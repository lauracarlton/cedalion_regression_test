# API Reference

Quick reference for the main functions in the cedalion regression test suite.

## Pipeline Functions

### STEP1: HRF Estimation

```python
from pipelines.STEP1_hrf_estimation import run_pipeline_estimate_hrf

hrf, mse = run_pipeline_estimate_hrf(
    ROOT_DIR="/path/to/data",
    TASK="BS",
    N_RUNS=3,
    NOISE_MODEL="ar_irls",  # or "ols"
    SAVE_DIR="/path/to/output",
    GOLDEN_CHANNEL='S53D28'
)
```

**Parameters:**
- `ROOT_DIR` (str): Path to BIDS-like dataset
- `TASK` (str): Task identifier in filenames
- `N_RUNS` (int): Number of runs per subject
- `NOISE_MODEL` (str): `"ols"` or `"ar_irls"`
- `SAVE_DIR` (str): Output directory (default: ROOT_DIR/tmp)
- `GOLDEN_CHANNEL` (str): Reference channel for testing

**Returns:**
- `hrf`: HRF estimates for golden channel (trial_type, chromo, time) in molar
- `mse`: Mean squared error (trial_type, chromo, time) in molar²

---

### STEP2: Channel-Space Group Average

```python
from pipelines.STEP2_get_group_average_channel import run_pipeline_channel_group_avg

results = run_pipeline_channel_group_avg(
    ROOT_DIR="/path/to/data",
    TASK="BS",
    NOISE_MODEL="ar_irls",
    REC_STR='conc',
    TRIAL_TYPES=['right', 'left'],
    GOLDEN_CHANNEL='S53D28'
)
```

**Parameters:**
- `ROOT_DIR` (str): Path to data with STEP1 outputs
- `TASK` (str): Task identifier
- `NOISE_MODEL` (str): Must match STEP1
- `REC_STR` (str): Recording type (`'conc'`)
- `TRIAL_TYPES` (list): Trial types to process
- `GOLDEN_CHANNEL` (str): Reference channel

**Returns (dict):**
- `'mean'`: Weighted mean HRF (trial_type, chromo, time)
- `'stderr'`: Standard error (trial_type, chromo, time)
- `'tstat'`: T-statistic (trial_type, chromo, time)
- `'mse_within'`: Within-subject MSE (trial_type, chromo, time)
- `'mse_btw'`: Between-subject MSE (trial_type, chromo, time)

---

### STEP3: Image Reconstruction

```python
from pipelines.STEP3_image_recon_on_HRF import run_pipeline_image_recon

cfg = {
    "method": "conc",           # 'conc' or 'mua2conc'
    "alpha_meas": 1e4,          # Measurement regularization
    "alpha_spatial": 1e-3,      # Spatial prior strength
    "lambda_R": 1e-6,           # Depth regularization
    "SB": True,                 # Use spatial basis functions
    "sigma_brain": 1,           # Brain basis width (mm)
    "sigma_scalp": 5            # Scalp basis width (mm)
}

image, post_mse = run_pipeline_image_recon(
    ROOT_DIR="/path/to/data",
    cfg=cfg,
    TASK="BS",
    NOISE_MODEL="ar_irls",
    REC_STR='conc',
    T_WIN=[5, 8]
)
```

**Parameters:**
- `ROOT_DIR` (str): Path to data
- `cfg` (dict): Reconstruction configuration (see above)
- `TASK` (str): Task identifier
- `NOISE_MODEL` (str): Must match STEP1
- `REC_STR` (str): Recording type
- `T_WIN` (list): Time window [start, end] in seconds for magnitude

**Returns:**
- `image`: Reconstructed image (trial_type, chromo, flatmap/vertex) in molar
- `post_mse`: Posterior MSE (trial_type, chromo, flatmap/vertex) in molar²

---

### STEP4: Image-Space Group Average

```python
from pipelines.STEP4_get_group_average_image import run_pipeline_image_group_avg

results = run_pipeline_image_group_avg(
    ROOT_DIR="/path/to/data",
    cfg=cfg,  # Same cfg as STEP3
    TASK="BS",
    NOISE_MODEL="ar_irls",
    TRIAL_TYPES=['right', 'left']
)
```

**Parameters:**
- `ROOT_DIR` (str): Path to data with STEP3 outputs
- `cfg` (dict): Must match STEP3 configuration
- `TASK` (str): Task identifier
- `NOISE_MODEL` (str): Must match STEP1
- `TRIAL_TYPES` (list): Trial types to aggregate

**Returns (dict):**
- `'mean'`: Group mean image (trial_type, chromo, flatmap/vertex)
- `'stderr'`: Standard error (trial_type, chromo, flatmap/vertex)
- `'tstat'`: T-statistic (trial_type, chromo, flatmap/vertex)
- `'mse_within'`: Within-subject MSE (trial_type, chromo, flatmap/vertex)
- `'mse_btw'`: Between-subject MSE (trial_type, chromo, flatmap/vertex)

---

## Processing Functions

### Channel Quality Control

```python
from modules.processing_func import prune_channels

rec, chs_pruned = prune_channels(
    rec,
    amp_thresh=[1e-3, 0.84] * units.V,
    sd_thresh=[0, 45] * units.mm,
    snr_thresh=5
)
```

**Parameters:**
- `rec` (dict): Recording with `'amp'` and `geo3d` fields
- `amp_thresh`: [min, max] amplitude in volts
- `sd_thresh`: [min, max] source-detector distance
- `snr_thresh`: Minimum signal-to-noise ratio

**Returns:**
- `rec`: Updated recording with `'amp_pruned'` field
- `chs_pruned`: Quality scores (0.0=bad, 0.4=good)

---

### GLM Analysis

```python
from modules.processing_func import GLM

results, hrf_estimate, hrf_mse = GLM(
    runs=[run1, run2, run3],
    cfg_GLM=cfg_GLM,
    geo3d=geo3d,
    pruned_chans_list=[pruned1, pruned2, pruned3],
    stim_list=[stim1, stim2, stim3]
)
```

**Parameters:**
- `runs` (list): List of concentration DataArrays (channel, chromo, time)
- `cfg_GLM` (dict): Configuration with keys:
  - `'t_pre'`, `'t_post'`: Time window (pint Quantity)
  - `'t_delta'`, `'t_std'`: Gaussian basis parameters
  - `'noise_model'`: `'ols'` or `'ar_irls'`
  - `'do_drift'`, `'do_drift_legendre'`: Drift flags
  - `'drift_order'`: Polynomial order
  - `'do_short_sep'`: Include short channels
  - `'distance_threshold'`: Short channel threshold
- `geo3d`: 3D optode geometry
- `pruned_chans_list` (list): Quality scores per run
- `stim_list` (list): Stimulus DataFrames (onset, duration, trial_type)

**Returns:**
- `results`: GLM results object (cedalion)
- `hrf_estimate`: HRF (channel, chromo, time, trial_type)
- `hrf_mse`: MSE (channel, chromo, time, trial_type)

---

### Design Matrix Components

#### Drift Regressors (Polynomial)
```python
from modules.processing_func import get_drift_regressors

drift_regressors = get_drift_regressors(runs, cfg_GLM)
```

#### Drift Regressors (Legendre)
```python
from modules.processing_func import get_drift_legendre_regressors

drift_regressors = get_drift_legendre_regressors(runs, cfg_GLM)
```

#### Short-Separation Regressors
```python
from modules.processing_func import get_short_regressors

ss_regressors = get_short_regressors(runs, pruned_chans_list, geo3d, cfg_GLM)
```

---

### Run Concatenation

```python
from modules.processing_func import concatenate_runs

Y_all, stim_df, runs_updated = concatenate_runs(
    runs=[run1, run2, run3],
    stim=[stim1, stim2, stim3]
)
```

**Parameters:**
- `runs` (list): Concentration DataArrays
- `stim` (list): Stimulus DataFrames

**Returns:**
- `Y_all`: Concatenated data (channel, chromo, time)
- `stim_df`: Concatenated stimulus DataFrame
- `runs_updated`: List of runs with updated time coordinates

---

### Spatial Smoothing

```python
from modules.processing_func import get_spatial_smoothing_kernel

kernel = get_spatial_smoothing_kernel(
    V_ras,
    sigma_mm=80 * units.mm
)
```

**Parameters:**
- `V_ras`: Vertex coordinates (n_vertices, 3) in mm
- `sigma_mm`: Gaussian kernel width

**Returns:**
- `kernel`: Smoothing weight matrix (n_vertices, n_vertices)

---

```python
from modules.processing_func import apply_spatial_smoothing_to_image

smoothed = apply_spatial_smoothing_to_image(image, kernel)
```

**Parameters:**
- `image`: Image DataArray (trial_type, chromo, flatmap/vertex)
- `kernel`: Smoothing kernel from `get_spatial_smoothing_kernel()`

**Returns:**
- `smoothed`: Spatially smoothed image

---

## Configuration Examples

### OLS Configuration
```python
cfg_GLM = {
    "do_drift": True,
    "do_drift_legendre": False,
    "do_short_sep": True,
    "drift_order": 3,
    "distance_threshold": 20 * units.mm,
    "short_channel_method": "mean",
    "noise_model": "ols",
    "t_delta": 1 * units.s,
    "t_std": 1 * units.s,
    "t_pre": 2 * units.s,
    "t_post": 10 * units.s,
}

cfg_bandpass = {
    "fmin": 0 * units.Hz,
    "fmax": 0.5 * units.Hz
}
```

### AR-IRLS Configuration
```python
cfg_GLM = {
    "do_drift": False,
    "do_drift_legendre": True,
    "do_short_sep": True,
    "drift_order": 3,
    "distance_threshold": 20 * units.mm,
    "short_channel_method": "mean",
    "noise_model": "ar_irls",
    "t_delta": 1 * units.s,
    "t_std": 1 * units.s,
    "t_pre": 2 * units.s,
    "t_post": 10 * units.s,
}

# No bandpass filtering for AR-IRLS
```

### Channel Pruning Configuration
```python
from cedalion import units

cfg_prune = {
    "snr_thresh": 5,
    "sd_thresh": [1, 40] * units.mm,
    "amp_thresh": [1e-5, 0.84] * units.V,
    "perc_time_clean_thresh": 0.6,
    "sci_threshold": 0.6,
    "psp_threshold": 0.1,
    "window_length": 5 * units.s,
    "flag_use_sci": False,
    "flag_use_psp": False,
}
```

### Image Reconstruction Configurations

#### Simple (No Spatial Basis)
```python
cfg_simple = {
    "method": "conc",           # Direct reconstruction
    "alpha_meas": 1e4,
    "alpha_spatial": 1e-3,
    "lambda_R": 1e-6,
    "SB": False,                # No spatial basis
    "sigma_brain": 1,           # Not used if SB=False
    "sigma_scalp": 5
}
```

#### With Spatial Basis Functions
```python
cfg_spatial_basis = {
    "method": "conc",
    "alpha_meas": 1e4,
    "alpha_spatial": 1e-2,      # Higher for spatial basis
    "lambda_R": 1e-6,
    "SB": True,                 # Use spatial basis
    "sigma_brain": 1,           # 1mm FWHM
    "sigma_scalp": 5            # 5mm FWHM
}
```

#### Indirect Reconstruction
```python
cfg_indirect = {
    "method": "mua2conc",       # Reconstruct absorption first
    "alpha_meas": 1e4,
    "alpha_spatial": 1e-3,
    "lambda_R": 1e-6,
    "SB": False,
    "sigma_brain": 1,
    "sigma_scalp": 5
}
```

---

## Data Structures

### Input: Amplitude Data
```python
amp : xr.DataArray
    Dimensions: (channel, wavelength, time)
    Coordinates:
        - channel: ['S1D1', 'S1D2', ...]
        - wavelength: [760, 850] (nm)
        - time: [0.0, 0.1, 0.2, ...] (s)
    Units: volts (V)
```

### Input: Events DataFrame
```python
events : pd.DataFrame
    Columns: ['onset', 'duration', 'trial_type']
    Example:
        onset  duration  trial_type
        10.5   2.0       right
        15.8   2.0       left
        21.2   2.0       right
```

### Input: Geometry
```python
geo3d : xr.DataArray
    Dimensions: (label, digitized)
    Coordinates:
        - label: ['S1', 'D1', 'S2', 'D2', ...]
        - digitized: [0, 1, 2] (x, y, z)
    Units: millimeters (mm)
```

### Output: HRF Estimates
```python
hrf : xr.DataArray
    Dimensions: (channel, chromo, time, trial_type)
    Coordinates:
        - channel: ['S1D1', 'S1D2', ...]
        - chromo: ['HbO', 'HbR']
        - time: [-2.0, -1.0, 0.0, ..., 10.0] (s, relative to onset)
        - trial_type: ['right', 'left']
    Units: molar (M, typically µM range)
```

### Output: Reconstructed Images
```python
image : xr.DataArray
    Dimensions: (trial_type, chromo, flatmap)
    Coordinates:
        - trial_type: ['right', 'left']
        - chromo: ['HbO', 'HbR']
        - flatmap: [0, 1, 2, ..., n_vertices-1]
    Units: molar (M)
```

---

## Common Workflows

### Full Channel-Space Analysis
```python
# Step 1: Estimate HRFs per subject
from pipelines.STEP1_hrf_estimation import run_pipeline_estimate_hrf

hrf, mse = run_pipeline_estimate_hrf(
    ROOT_DIR="/data/BS_bids",
    TASK="BS",
    N_RUNS=3,
    NOISE_MODEL="ar_irls",
    SAVE_DIR="/data/output"
)

# Step 2: Compute group statistics
from pipelines.STEP2_get_group_average_channel import run_pipeline_channel_group_avg

group_results = run_pipeline_channel_group_avg(
    ROOT_DIR="/data/BS_bids",
    TASK="BS",
    NOISE_MODEL="ar_irls"
)

# Access results
mean_hrf = group_results['mean']
tstat = group_results['tstat']
```

### Full Image-Space Analysis
```python
# Step 1: Estimate HRFs (same as above)
# ...

# Step 3: Reconstruct images per subject
from pipelines.STEP3_image_recon_on_HRF import run_pipeline_image_recon

cfg = {
    "method": "conc",
    "alpha_meas": 1e4,
    "alpha_spatial": 1e-3,
    "lambda_R": 1e-6,
    "SB": True,
    "sigma_brain": 1,
    "sigma_scalp": 5
}

image, post_mse = run_pipeline_image_recon(
    ROOT_DIR="/data/BS_bids",
    cfg=cfg,
    TASK="BS",
    NOISE_MODEL="ar_irls",
    T_WIN=[5, 8]
)

# Step 4: Compute group statistics
from pipelines.STEP4_get_group_average_image import run_pipeline_image_group_avg

group_images = run_pipeline_image_group_avg(
    ROOT_DIR="/data/BS_bids",
    cfg=cfg,
    TASK="BS",
    NOISE_MODEL="ar_irls"
)

# Access results
mean_image = group_images['mean']
tstat_map = group_images['tstat']
```

---

## Units Reference

Common units used throughout the codebase:

```python
from cedalion import units

# Time
t = 10 * units.s          # seconds
f = 0.5 * units.Hz        # hertz

# Distance
d = 40 * units.mm         # millimeters

# Amplitude
a = 0.5 * units.V         # volts

# Concentration
c = 5 * units.micromolar  # µM (micromolar)
c = 5e-6 * units.molar    # M (molar, equivalent)

# Optical density
od = 0.1 * units.AU       # arbitrary units

# Absorption
mua = 0.01 / units.mm     # 1/mm
```

### Unit conversions:
```python
# Convert between units
c_molar = c_micromolar.to('molar')

# Quantify/dequantify xarray
data_with_units = data.pint.quantify('molar')
data_without_units = data.pint.dequantify()

# Check units
print(c.units)  # micromolar
```

---

## Error Handling

### Common errors:

#### Unit mismatch:
```python
# Error
DimensionalityError: Cannot convert from 'volt' to 'millimeter'

# Fix: ensure consistent units
amp_thresh = [1e-3, 0.84] * units.V
sd_thresh = [1, 40] * units.mm
```

#### Dimension mismatch:
```python
# Error
ValueError: operands could not be broadcast together

# Fix: use explicit dimension names
result = xr.dot(A, B, dims='channel')  # Specify reduction dim
```

#### Missing data:
```python
# Error
KeyError: 'amp_pruned'

# Fix: ensure prune_channels was called first
rec, chs_pruned = prune_channels(rec, ...)
# Now rec['amp_pruned'] exists
```

---

## Performance Tips

1. **Load data once:** Don't reload geometry/forward model in loops
2. **Vectorize:** Use xarray operations instead of loops
3. **Chunk large datasets:** Use dask for out-of-core computation
4. **Profile first:** Use cProfile before optimizing

---

For complete examples, see the test files in `tests/`.
