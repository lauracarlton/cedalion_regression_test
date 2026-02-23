# cedalion_regression_test
A regression testing suite for fNIRS (functional Near-Infrared Spectroscopy) data analysis pipelines using the Cedalion library. This repository provides end-to-end validation of channel-space and image-space analyses, including preprocessing, HRF (Hemodynamic Response Function) estimation, and group-level statistical testing.

## Overview

This project implements automated regression tests for fNIRS analysis workflows, ensuring reproducibility and detecting regressions in:
- **Channel-space analysis**: Single-channel and group-level HRF estimation
- **Image-space analysis**: Brain imaging reconstruction with multiple regularization schemes
- **Statistical analysis**: Within-subject and between-subject variance estimation, group averaging, t-statistics

The pipeline processes BIDS-like fNIRS datasets through multiple stages, comparing outputs against validated reference data to ensure numerical stability across software updates.

## Project Structure

```
cedalion_regression_test/
├── README.md                          # This file
├── modules/
│   └── processing_func.py            # Core processing functions (GLM, pruning, spatial smoothing)
├── pipelines/
│   ├── STEP1_hrf_estimation.py       # Per-subject HRF estimation via GLM
│   ├── STEP2_get_group_average_channel.py  # Group-level channel-space statistics
│   ├── STEP3_image_recon_on_HRF.py   # Image reconstruction on HRF estimates
│   └── STEP4_get_group_average_image.py    # Group-level image-space statistics
└── tests/
    ├── test_hrf_estimation.py        # Regression tests for channel-space pipeline
    ├── test_image_recon.py           # Regression tests for image-space pipeline
    └── data/                         # Test data and reference outputs
        ├── probe/                    # Forward models and geometry
        ├── reference/                # Golden reference outputs
        ├── sub-*/                    # Subject data (SNIRF format)
        └── tmp/                      # Intermediate outputs

```

## Key Components

### Modules (`modules/`)

**`processing_func.py`**: Core data processing and analysis utilities

Key functions:
- **Channel quality control**
  - `prune_channels()`: Flag poor-quality channels based on amplitude, SNR, and SD distance
  - `prune_mask_ts()`: Mask pruned channels with NaN values
  
- **GLM analysis**
  - `GLM()`: Fit hemodynamic response functions using OLS or AR-IRLS noise models
  - `estimate_HRF_from_beta()`: Reconstruct HRF from GLM beta coefficients via Gaussian basis functions
  - `estimate_HRF_cov()`: Compute HRF uncertainty from beta covariance matrix
  
- **Design matrix construction**
  - `get_drift_regressors()`: Polynomial drift regressors for OLS
  - `get_drift_legendre_regressors()`: Orthogonal Legendre drift for AR-IRLS
  - `get_short_regressors()`: Short-separation channel regressors for systemic noise removal
  - `concatenate_runs()`: Merge multiple runs for joint GLM fitting
  
- **Spatial smoothing**
  - `get_spatial_smoothing_kernel()`: Gaussian smoothing for surface mesh
  - `apply_spatial_smoothing_to_image()`: Smooth reconstructed brain images

### Pipelines (`pipelines/`)

#### STEP 1: HRF Estimation (`STEP1_hrf_estimation.py`)
Performs per-subject preprocessing and GLM-based HRF estimation.

**Inputs:**
- BIDS-like subject folders with data timeseries netcdf files and `_events.tsv` stimulus timing
- Forward model (`Adot.nc`) and geometry (`geo3d.pkl`)

**Configurable parameters:**
- `NOISE_MODEL`: `'ols'` or `'ar_irls'` (controls drift and filtering)
- `N_RUNS`: Number of runs per subject
- `TASK`: Task identifier
- Channel pruning thresholds (SNR, amplitude, SD distance)

**Outputs:**
- Per-subject HRF estimates in concentration space (µM)
- Per-channel MSE (uncertainty) estimates
- Bad channel indices

**Key processing steps:**
1. Load amplitude data and apply median filtering
2. Prune channels based on quality metrics
3. Convert to optical density, apply TDDR motion correction (OLS only)
4. Bandpass filter (OLS) or skip filtering (AR-IRLS)
5. Convert to concentration (HbO/HbR)
6. Fit GLM with drift and short-separation regressors
7. Estimate HRF and MSE for each trial type

#### STEP 2: Group Average (Channel Space) (`STEP2_get_group_average_channel.py`)
Computes weighted group averages across subjects for channel-space data.

**Inputs:**
- Per-subject HRF pickle files from STEP 1

**Outputs:**
- Weighted mean HRF across subjects
- Within-subject MSE (measurement noise)
- Between-subject MSE (inter-subject variability)
- Standard error and t-statistics

#### STEP 3: Image Reconstruction (`STEP3_image_recon_on_HRF.py`)
Reconstructs brain images from channel-space HRF estimates.

**Inputs:**
- Per-subject HRF estimates from STEP 1
- Forward model sensitivity matrix (`Adot`)
- ICBM152 head model (brain/scalp surfaces)

**Configurable parameters:**
```python
cfg = {
    "method": 'conc' or 'mua2conc',  # Direct vs indirect reconstruction
    "alpha_meas": 1e4,                # Measurement noise regularization
    "alpha_spatial": 1e-3,            # Spatial prior strength
    "lambda_R": 1e-6,                 # Depth regularization
    "SB": True/False,                 # Use spatial basis functions
    "sigma_brain": 1,                 # Brain basis width (mm)
    "sigma_scalp": 5                  # Scalp basis width (mm)
}
```

**Reconstruction methods:**
- **Direct (`conc`)**: Reconstruct concentration changes directly
- **Indirect (`mua2conc`)**: Reconstruct absorption coefficient, then convert to concentration

**Outputs:**
- Per-subject reconstructed images (µM)
- Per-voxel MSE estimates

#### STEP 4: Group Average (Image Space) (`STEP4_get_group_average_image.py`)
Computes weighted group averages for image-space data.

**Inputs:**
- Per-subject image reconstructions from STEP 3

**Outputs:**
- Weighted mean group image
- Within-subject and between-subject MSE
- Standard error and t-statistic maps

### Tests (`tests/`)

#### `test_hrf_estimation.py`
Regression tests for channel-space pipeline (STEP 1-2).

**Tests:**
1. Single-subject HRF estimation for single channel
2. Single-subject MSE estimation
3. Group-level mean, standard error, t-statistics
4. Within-subject and between-subject MSE

#### `test_image_recon.py`
Regression tests for image-space pipeline (STEP 3-4).

**Tests multiple configurations:**
- Direct vs indirect reconstruction
- With/without spatial basis functions
- Different regularization parameters

Each test compares outputs against reference data with tolerances:
- HRF values: rtol=1e-3, atol=1e-8
- MSE values: rtol=1e-3, atol=1e-12
- T-statistics: rtol=1e-4, atol=1e-8

## Installation & Dependencies

### Required packages:
cedalion - see cedalion Git Repo for more details on the installation

## Usage

### Running regression tests:

```bash
# Test channel-space pipeline
cd tests
python test_hrf_estimation.py

# Test image-space pipeline
python test_image_recon.py
```

### Running individual pipeline steps:

```python
from pipelines.STEP1_hrf_estimation import run_pipeline_estimate_hrf

# Estimate HRFs for all subjects
hrf, mse = run_pipeline_estimate_hrf(
    ROOT_DIR="/path/to/data",
    TASK="BS",
    N_RUNS=3,
    NOISE_MODEL="ar_irls",
    SAVE_DIR="/path/to/output",
    GOLDEN_CHANNEL='S53D28'
)
```

## Configuration Guidelines

### Noise Model Selection

**OLS (Ordinary Least Squares):**
- Requires TDDR motion correction
- Uses polynomial drift regressors
- Applies bandpass filtering (0-0.5 Hz)
- Faster computation
- Best for clean data with minimal autocorrelation

**AR-IRLS (Autoregressive Iteratively Reweighted Least Squares):**
- No motion correction applied
- Uses Legendre polynomial drift
- No bandpass filtering
- Robust to colored noise and autocorrelation
- Recommended for noisy data

### Channel Pruning Thresholds

Default values:
```python
cfg_prune = {
    "snr_thresh": 5,                    # Signal-to-noise ratio
    "sd_thresh": [1, 40] * units.mm,    # Source-detector distance range
    "amp_thresh": [1e-5, 0.84] * units.V # Amplitude range
}
```

### Image Reconstruction Parameters

- **alpha_meas**: Higher values (1e4-1e6) = more regularization, smoother images
- **alpha_spatial**: Controls spatial prior strength (1e-3 to 1e-2)
- **lambda_R**: Depth weighting to overcome superficial bias (1e-6 typical)
- **Spatial basis functions**: Reduce dimensionality, improve conditioning

## Reference Data

Golden reference outputs are stored in `tests/data/reference/` as NetCDF files (`.nc`):

**Channel-space:**
- `single_channel_single_subj_hrf.nc`: Single subject HRF for validation channel
- `single_channel_single_subj_hrf_mse.nc`: Corresponding MSE
- `single_channel_group_*.nc`: Group statistics (mean, stderr, tstat, mse_btw, mse_within)

**Image-space:**
- `single_sub_image_hrf_mag_*.nc`: Per-subject reconstructed images
- `single_sub_post_hrf_mag_*.nc`: Per-subject posterior MSE
- `image_group_*.nc`: Group statistics for multiple regularization schemes

## Testing Workflow

1. **Generate reference data** (first run):
   - Tests run pipelines and save outputs as reference
   
2. **Regression testing** (subsequent runs):
   - Tests compare new outputs against saved reference
   - `assert_xarray_close()` with specified tolerances
   - Fails if outputs deviate beyond tolerance
   
3. **Reference updates**:
   - If changes are intentional and validated, tests update reference files
   - Only updates on successful comparison (manual verification needed for failures)

## GLM Design Matrix

The GLM design matrix combines multiple regressor types:

```
Y(t) = β_HRF * HRF(t) + β_drift * Drift(t) + β_short * Short(t) + ε(t)
```

Where:
- **HRF regressors**: Gaussian basis functions convolved with stimulus timing
- **Drift regressors**: Polynomial (OLS) or Legendre (AR-IRLS) basis
- **Short-separation regressors**: Mean of channels with SD < 20mm
- **ε(t)**: Residual noise (assumed white for OLS, AR for AR-IRLS)

## Statistical Analysis

### Weighted Group Averaging

For subjects i=1...N with measurements X_i and MSE σ²_i:

**Within-subject variance:**
```
σ²_within = 1 / Σ(1/σ²_i)
```

**Between-subject variance:**
```
σ²_between = Σ[(X_i - X_weighted)² / σ²_i] / N × σ²_within
```

**Weighted mean:**
```
X_weighted = Σ(X_i / (σ²_within + σ²_between)) / Σ(1 / (σ²_within + σ²_between))
```

**Standard error:**
```
SE = √(1 / Σ(1 / (σ²_within + σ²_between)))
```

**T-statistic:**
```
t = X_weighted / SE
```

## Data Format

### Input data (BIDS-like):
```
ROOT_DIR/
├── sub-586/
│   └── nirs/
│       ├── sub-586_task-BS_run-01_samples.nc
│       ├── sub-586_task-BS_run-01_nirs_events.tsv
│       └── ...
├── probe/
│   ├── Adot.nc           # Forward model sensitivity
│   └── geo3d.pkl         # 3D optode geometry
```

### Stimulus timing format (`_events.tsv`):
```
onset    duration    trial_type
10.5     2.0         right
15.8     2.0         left
...
```

## Author

**Laura Carlton** (lcarlton@bu.edu)  
Created: January 16, 2025

