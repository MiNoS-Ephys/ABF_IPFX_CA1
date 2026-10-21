# Reproducible ABF current-clamp analysis with IPFX

Python pipeline for protocol-specific analysis of ABF current-clamp recordings using **pyABF** and the Allen Institute **IPFX** spike feature extractor.

## What this repository does

The analysis assumes a 11-sweeps current-clamp protocol with a 1 s stimulus beginning at 90.6 ms. Current starts at 0 pA and increases by 30 pA per sweep (0–300 pA). The target sweep for the dedicated spike analysis is 150 pA.

The script extracts spike features with IPFX, computes ISIs, firing-rate/F-I information, rheobase-related summary values, and exports the results to one Excel workbook.

### 150 pA additions

In addition to the original spike-by-spike output, the workbook contains:

- `Sweep_150pA_Spikes`: every detected spike in the 150 pA sweep.
- `150pA_Spike_Means`: one row per ABF file with the mean of the numeric spike metrics across all spikes detected at 150 pA, plus `n_spikes_averaged`.
- `150pA_First_Spike`: one row per ABF file containing only the first detected spike (`stim_spike_number == 1`) from the 150 pA sweep.
- `Sweep_150pA_ISIs`: ISIs from the 150 pA sweep.

Because each ABF file is retained separately by filename, paired control/treatment recordings remain separate rows and can be matched using your experimental naming convention.

## Repository structure

```text
.
├── analyze_abf_ipfx.py      # Main analysis script
├── README.md                # Documentation
├── requirements.txt         # Python dependencies
├── environment.yml          # Conda environment
├── CITATION.cff             # Citation metadata (edit placeholders before release)
├── LICENSE                  # MIT template (edit author/year placeholder)
├── .gitignore
├── data/
│   └── README.md            # Put ABF input files here locally
└── results/
    └── README.md            # Generated output location
```

## Installation

### Option A: conda

```bash
conda env create -f environment.yml
conda activate abf-ipfx
```

### Option B: pip

Create and activate a Python virtual environment, then run:

```bash
pip install -r requirements.txt
```

## Running the analysis

Put your `.abf` files in `data/`, then from the repository root run:

```bash
python analyze_abf_ipfx.py data --output results
```

The script writes:

```text
results/ABF_analysis_IPFX.xlsx
```

You can also analyze a single ABF file:

```bash
python analyze_abf_ipfx.py data/example.abf --output results
```

## Excel output

The workbook contains these sheets:

- `Cell_Summary`
- `Sweep_Summary`
- `All_Spikes_IPFX`
- `All_ISIs`
- `FI_Curve`
- `Sweep_150pA_Spikes`
- `150pA_Spike_Means`
- `150pA_First_Spike`
- `Sweep_150pA_ISIs`
- `Methods`

## Important protocol assumptions

The constants near the top of `analyze_abf_ipfx.py` define the protocol:

```python
STIM_START_S = 0.0906
STIM_DURATION_S = 1.0
CURRENT_START_PA = 0.0
CURRENT_STEP_PA = 30.0
EXPECTED_SWEEPS = 16
TARGET_CURRENT_PA = 150.0
```

If your acquisition protocol changes, review these values before analyzing new data.

## Raw data and GitHub

ABF files are ignored by the included `.gitignore` so raw electrophysiology data are not accidentally committed. Keep raw data in an appropriate archive/repository and document its location if the analysis is published.

## Reproducibility notes

The pipeline records its feature definitions in the workbook's `Methods` sheet. IPFX parameters can also be changed from the command line; run:

```bash
python analyze_abf_ipfx.py --help
```

Before publishing the repository, edit the placeholders in `CITATION.cff` and `LICENSE`, and add the final repository/archive URL and release metadata.
