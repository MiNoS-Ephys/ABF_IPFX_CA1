#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyabf
try:
    from ipfx.feature_extractor import SpikeFeatureExtractor
except ImportError as exc:
    raise SystemExit("IPFX is not installed. In Jupyter run: !pip install ipfx") from exc

STIM_START_S=0.0906
STIM_DURATION_S=1.0
STIM_END_S=STIM_START_S+STIM_DURATION_S
CURRENT_START_PA=0.0
CURRENT_STEP_PA=30.0
EXPECTED_SWEEPS=16
TARGET_CURRENT_PA=150.0

def build_current_trace(t,current_pa):
    i=np.zeros_like(t,dtype=float)
    i[(t>=STIM_START_S)&(t<STIM_END_S)]=current_pa
    return i

def safe_mean(df,col):
    if df.empty or col not in df.columns: return np.nan
    x=pd.to_numeric(df[col],errors='coerce')
    return float(x.mean()) if x.notna().any() else np.nan

def normalize_ipfx_spikes(df):
    df=df.copy()
    mapping={'threshold_t':'threshold_time_s','threshold_v':'threshold_voltage_mV','peak_t':'peak_time_s','peak_v':'peak_voltage_mV','fast_trough_t':'IPFX_fast_trough_time_s','fast_trough_v':'IPFX_fast_trough_voltage_mV','upstroke':'max_rise_slope_V_per_s','downstroke':'max_decay_slope_V_per_s'}
    for old,new in mapping.items():
        if old in df.columns: df[new]=df[old]
    df['half_width_ms']=pd.to_numeric(df['width'],errors='coerce')*1000 if 'width' in df.columns else np.nan
    if {'threshold_voltage_mV','peak_voltage_mV'}.issubset(df.columns):
        df['spike_amplitude_mV']=pd.to_numeric(df['peak_voltage_mV'],errors='coerce')-pd.to_numeric(df['threshold_voltage_mV'],errors='coerce')
    else: df['spike_amplitude_mV']=np.nan
    if 'IPFX_fast_trough_voltage_mV' in df.columns:
        df['fAHP_like_voltage_mV_IPFX_fast_trough']=df['IPFX_fast_trough_voltage_mV']
        if 'threshold_voltage_mV' in df.columns:
            df['fAHP_like_amplitude_mV_from_threshold']=pd.to_numeric(df['threshold_voltage_mV'],errors='coerce')-pd.to_numeric(df['IPFX_fast_trough_voltage_mV'],errors='coerce')
        else: df['fAHP_like_amplitude_mV_from_threshold']=np.nan
    else:
        df['fAHP_like_voltage_mV_IPFX_fast_trough']=np.nan
        df['fAHP_like_amplitude_mV_from_threshold']=np.nan
    return df


def add_ahp_duration(spikes, t, v):
    """
    Add AHP duration for each spike.

    Definition: time from the IPFX fast-trough time to the first subsequent
    upward crossing of that spike's threshold voltage. The search is limited
    to the earlier of the next spike threshold or the stimulus end.

    Returns milliseconds. NaN means no valid recovery crossing was found.
    """
    if spikes.empty:
        spikes['AHP_duration_ms'] = np.nan
        return spikes

    spikes = spikes.copy()
    durations = []
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)

    for idx, row in spikes.iterrows():
        trough_t = row.get('IPFX_fast_trough_time_s', np.nan)
        threshold_v = row.get('threshold_voltage_mV', np.nan)

        if not np.isfinite(trough_t) or not np.isfinite(threshold_v):
            durations.append(np.nan)
            continue

        start_idx = int(np.searchsorted(t, trough_t, side='left'))

        # Avoid crossing into the next action potential when possible.
        if idx + 1 < len(spikes) and 'threshold_time_s' in spikes.columns:
            next_thr_t = pd.to_numeric(spikes.iloc[idx + 1]['threshold_time_s'], errors='coerce')
            stop_t = min(float(next_thr_t), STIM_END_S) if np.isfinite(next_thr_t) else STIM_END_S
        else:
            stop_t = STIM_END_S

        stop_idx = int(np.searchsorted(t, stop_t, side='right'))
        stop_idx = min(stop_idx, len(t))

        if start_idx >= stop_idx - 1:
            durations.append(np.nan)
            continue

        seg_v = v[start_idx:stop_idx]
        crossings = np.flatnonzero(
            (seg_v[:-1] < threshold_v) & (seg_v[1:] >= threshold_v)
        )

        if len(crossings) == 0:
            durations.append(np.nan)
            continue

        j = start_idx + int(crossings[0])
        v1, v2 = v[j], v[j + 1]
        t1, t2 = t[j], t[j + 1]
        if v2 == v1:
            crossing_t = t2
        else:
            frac = (threshold_v - v1) / (v2 - v1)
            crossing_t = t1 + frac * (t2 - t1)

        duration_ms = (crossing_t - trough_t) * 1000.0
        durations.append(float(duration_ms) if duration_ms >= 0 else np.nan)

    spikes['AHP_duration_ms'] = durations
    return spikes


def analyze_abf(path,args):
    abf=pyabf.ABF(str(path))
    if abf.sweepCount!=EXPECTED_SWEEPS:
        print(f'WARNING: {path.name} has {abf.sweepCount} sweeps; expected {EXPECTED_SWEEPS}.')
    spike_tables=[]; isi_tables=[]; sweep_rows=[]
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        t=np.asarray(abf.sweepX,float); v=np.asarray(abf.sweepY,float)
        current_pa=CURRENT_START_PA+CURRENT_STEP_PA*int(sweep)
        i=build_current_trace(t,current_pa)
        fs=float(abf.dataRate); nyq_khz=fs/2000.0
        filt=args.filter_khz
        if filt is not None and filt>=nyq_khz: filt=None
        ext=SpikeFeatureExtractor(start=STIM_START_S,end=STIM_END_S,filter=filt,dv_cutoff=args.dv_cutoff,max_interval=args.max_interval_s,min_height=args.min_height_mv,min_peak=args.min_peak_mv,thresh_frac=args.thresh_frac)
        try: raw=ext.process(t=t,v=v,i=i)
        except Exception as e:
            print(f'ERROR {path.name} sweep {sweep}: {e}'); raw=pd.DataFrame()
        spikes=normalize_ipfx_spikes(raw)
        spikes=add_ahp_duration(spikes,t,v)
        if not spikes.empty:
            spikes.insert(0,'stim_spike_number',np.arange(1,len(spikes)+1))
            spikes.insert(0,'current_pA',current_pa); spikes.insert(0,'sweep',int(sweep)); spikes.insert(0,'file',path.name)
            if 'peak_time_s' in spikes.columns: spikes['latency_from_stim_ms']=(pd.to_numeric(spikes['peak_time_s'],errors='coerce')-STIM_START_S)*1000
            else: spikes['latency_from_stim_ms']=np.nan
            spike_tables.append(spikes)
        n=len(spikes)
        pts=pd.to_numeric(spikes['peak_time_s'],errors='coerce').dropna().to_numpy() if n and 'peak_time_s' in spikes.columns else np.array([])
        isi=np.diff(pts)*1000 if len(pts)>=2 else np.array([])
        if len(isi):
            isi_tables.append(pd.DataFrame({'file':path.name,'sweep':int(sweep),'current_pA':current_pa,'ISI_number':np.arange(1,len(isi)+1),'from_spike':np.arange(1,len(isi)+1),'to_spike':np.arange(2,len(isi)+2),'ISI_ms':isi}))
        sweep_rows.append({'file':path.name,'sweep':int(sweep),'current_pA':current_pa,'n_spikes_during_stim':int(n),'firing_rate_Hz':float(n/STIM_DURATION_S),'first_spike_latency_ms':float((pts[0]-STIM_START_S)*1000) if len(pts) else np.nan,'mean_threshold_voltage_mV_IPFX':safe_mean(spikes,'threshold_voltage_mV'),'mean_peak_voltage_mV_IPFX':safe_mean(spikes,'peak_voltage_mV'),'mean_spike_amplitude_mV':safe_mean(spikes,'spike_amplitude_mV'),'mean_half_width_ms_IPFX':safe_mean(spikes,'half_width_ms'),'mean_max_rise_slope_V_per_s_IPFX':safe_mean(spikes,'max_rise_slope_V_per_s'),'mean_max_decay_slope_V_per_s_IPFX':safe_mean(spikes,'max_decay_slope_V_per_s'),'mean_fast_trough_voltage_mV_IPFX':safe_mean(spikes,'IPFX_fast_trough_voltage_mV'),'mean_fAHP_like_voltage_mV_IPFX_fast_trough':safe_mean(spikes,'fAHP_like_voltage_mV_IPFX_fast_trough'),'mean_fAHP_like_amplitude_mV_from_threshold':safe_mean(spikes,'fAHP_like_amplitude_mV_from_threshold'),'mean_upstroke_downstroke_ratio_IPFX':safe_mean(spikes,'upstroke_downstroke_ratio'),'mean_AHP_duration_ms':safe_mean(spikes,'AHP_duration_ms'),'n_ISI':len(isi),'mean_ISI_ms':float(np.mean(isi)) if len(isi) else np.nan,'median_ISI_ms':float(np.median(isi)) if len(isi) else np.nan,'min_ISI_ms':float(np.min(isi)) if len(isi) else np.nan,'max_ISI_ms':float(np.max(isi)) if len(isi) else np.nan,'first_ISI_ms':float(isi[0]) if len(isi) else np.nan,'last_ISI_ms':float(isi[-1]) if len(isi) else np.nan})
    return (pd.concat(spike_tables,ignore_index=True) if spike_tables else pd.DataFrame(),pd.DataFrame(sweep_rows),pd.concat(isi_tables,ignore_index=True) if isi_tables else pd.DataFrame(columns=['file','sweep','current_pA','ISI_number','from_spike','to_spike','ISI_ms']))


def summarize_target_spikes(target_spikes):
    """One row per ABF file: mean of numeric spike metrics across all spikes at TARGET_CURRENT_PA."""
    if target_spikes.empty:
        return pd.DataFrame(columns=['file','current_pA','n_spikes_averaged'])
    id_cols={'file','sweep','current_pA','stim_spike_number'}
    metric_cols=[c for c in target_spikes.columns if c not in id_cols and pd.api.types.is_numeric_dtype(target_spikes[c])]
    rows=[]
    for fn,g in target_spikes.groupby('file',sort=False):
        row={'file':fn,'current_pA':TARGET_CURRENT_PA,'n_spikes_averaged':int(len(g))}
        for c in metric_cols:
            row['mean_'+c]=safe_mean(g,c)
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_first_target_spike(target_spikes):
    """One row per ABF file: metrics from only spike #1 of the TARGET_CURRENT_PA sweep."""
    if target_spikes.empty:
        return pd.DataFrame(columns=['file','sweep','current_pA','stim_spike_number'])
    first=target_spikes[target_spikes['stim_spike_number']==1].copy()
    # Keep the same spike-level columns so each file has its first 150-pA spike directly available.
    return first.reset_index(drop=True)


def summarize_cells(sweep_df):
    rows=[]
    for fn,g in sweep_df.groupby('file',sort=False):
        firing=g[g.n_spikes_during_stim>0]
        if len(firing):
            rheo=float(firing.current_pA.min()); first=int(firing.loc[firing.current_pA.idxmin(),'sweep'])
        else: rheo=np.nan; first=np.nan
        imax=g.firing_rate_Hz.idxmax()
        rows.append({'file':fn,'rheobase_pA':rheo,'first_firing_sweep':first,'max_firing_rate_Hz':float(g.loc[imax,'firing_rate_Hz']),'current_at_max_rate_pA':float(g.loc[imax,'current_pA']),'total_stimulus_spikes':int(g.n_spikes_during_stim.sum())})
    return pd.DataFrame(rows)

def autofit(writer,name,df):
    ws=writer.book[name]; ws.freeze_panes='A2'; ws.auto_filter.ref=ws.dimensions
    for j,c in enumerate(df.columns,1):
        vals=[str(c)]+([] if df.empty else df[c].astype(str).head(500).tolist())
        ws.column_dimensions[ws.cell(1,j).column_letter].width=min(max(map(len,vals))+2,38)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('input'); p.add_argument('--output',default='ipfx_results')
    p.add_argument('--filter-khz',type=float,default=10.0); p.add_argument('--dv-cutoff',type=float,default=20.0)
    p.add_argument('--max-interval-s',type=float,default=0.005); p.add_argument('--min-height-mv',type=float,default=2.0)
    p.add_argument('--min-peak-mv',type=float,default=-30.0); p.add_argument('--thresh-frac',type=float,default=0.05)
    args=p.parse_args()
    src=Path(args.input).expanduser().resolve(); out=Path(args.output).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    files=[src] if src.is_file() else sorted(src.glob('*.abf'))
    if not files: raise SystemExit('No .abf files found.')
    S=[]; W=[]; I=[]
    for f in files:
        print(f'Analyzing {f.name} with IPFX...')
        s,w,i=analyze_abf(f,args); S.append(s); W.append(w); I.append(i)
    spikes=pd.concat(S,ignore_index=True) if S else pd.DataFrame(); sweeps=pd.concat(W,ignore_index=True); isis=pd.concat(I,ignore_index=True) if I else pd.DataFrame()
    cells=summarize_cells(sweeps); fi=sweeps[['file','sweep','current_pA','n_spikes_during_stim','firing_rate_Hz']].copy()
    target_spikes=spikes[spikes.current_pA==TARGET_CURRENT_PA].copy() if not spikes.empty else pd.DataFrame()
    target_isis=isis[isis.current_pA==TARGET_CURRENT_PA].copy() if not isis.empty else pd.DataFrame()
    target_spikes_mean=summarize_target_spikes(target_spikes)
    target_first_spike=summarize_first_target_spike(target_spikes)
    methods=pd.DataFrame({'item':['Spike extractor','ABF reader','Stimulus','Current protocol','fAHP-like definition','AHP duration definition'],'value':['IPFX SpikeFeatureExtractor','pyABF',f'{STIM_START_S}-{STIM_END_S} s','0-450 pA, +30 pA/sweep','IPFX fast_trough_v; labeled explicitly as fast trough','Time from IPFX fast_trough_t to first subsequent upward crossing of that spike threshold_v, limited to before next spike threshold or stimulus end']})
    xp=out/'ABF_analysis_IPFX.xlsx'
    with pd.ExcelWriter(xp,engine='openpyxl') as writer:
        for name,df in {'Cell_Summary':cells,'Sweep_Summary':sweeps,'All_Spikes_IPFX':spikes,'All_ISIs':isis,'FI_Curve':fi,'Sweep_150pA_Spikes':target_spikes,'150pA_Spike_Means':target_spikes_mean,'150pA_First_Spike':target_first_spike,'Sweep_150pA_ISIs':target_isis,'Methods':methods}.items():
            df.to_excel(writer,sheet_name=name,index=False); autofit(writer,name,df)
    print(f'Done. Excel saved to: {xp}')

if __name__=='__main__': main()
