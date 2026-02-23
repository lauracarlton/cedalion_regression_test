# Testing Guide

This document provides detailed information about the regression testing framework for the cedalion fNIRS analysis pipeline.

## Overview

The testing suite validates two main analysis pathways:
1. **Channel-space analysis** (`test_hrf_estimation.py`)
2. **Image-space analysis** (`test_image_recon.py`)

Each test compares pipeline outputs against validated reference data to detect regressions introduced by code changes or dependency updates.

## Test Data Structure

```
tests/data/
├── probe/
│   ├── Adot.nc              # Forward model sensitivity matrix
│   └── geo3d.pkl            # 3D optode geometry
├── reference/
│   ├── single_channel_*.nc  # Channel-space reference outputs
│   └── image_*.nc           # Image-space reference outputs
├── sub-586/                 # Subject 1 fNIRS data
│   └── nirs/
│       ├── sub-586_task-BS_run-01_nirs_sample.nc
│       ├── sub-586_task-BS_run-01_nirs_events.tsv
│       └── ...
├── sub-587/                 # Subject 2
├── sub-592/                 # Subject 3
├── sub-618/                 # Subject 4
├── sub-621/                 # Subject 5
└── tmp/                     # Intermediate outputs (created during tests)
```

## Test Suite Details

### Test 1: Channel-Space HRF Estimation (`test_hrf_estimation.py`)

**Purpose:** Validate single-subject and group-level HRF estimation in channel space.

**Pipeline steps tested:**
1. **STEP1**: Per-subject preprocessing and GLM
2. **STEP2**: Group-level weighted averaging

**Tests performed:**

#### 1.1 Single-Subject HRF
- **Function:** `run_pipeline_estimate_hrf()`
- **Golden channel:** `S53D28` (single channel for regression testing)
- **Output:** HRF estimates (µM) with dimensions (trial_type, chromo, time)
- **Reference:** `single_channel_single_subj_hrf.nc`
- **Tolerance:** rtol=1e-3, atol=1e-8

#### 1.2 Single-Subject MSE
- **Output:** Mean squared error for HRF
- **Reference:** `single_channel_single_subj_hrf_mse.nc`
- **Tolerance:** rtol=1e-3, atol=1e-12

#### 1.3 Group-Level Statistics
- **Function:** `run_pipeline_channel_group_avg()`
- **Outputs tested:**
  - `mean`: Weighted group average HRF
  - `stderr`: Standard error
  - `tstat`: T-statistic
  - `mse_within`: Within-subject variance
  - `mse_btw`: Between-subject variance
- **References:** `single_channel_group_*.nc`
- **Tolerances:**
  - Mean/stderr: rtol=1e-3, atol=1e-8
  - MSE: rtol=1e-3, atol=1e-12
  - T-stat: rtol=1e-4, atol=1e-8

### Test 2: Image-Space Reconstruction (`test_image_recon.py`)

**Purpose:** Validate DOT image reconstruction and group averaging.

**Pipeline steps tested:**
1. **STEP3**: Per-subject image reconstruction from HRF
2. **STEP4**: Group-level image statistics

**Reconstruction configurations tested:**

```python
cfg_list = [
    # Config 1: Indirect, no spatial basis
    {"alpha_meas": 1e4, "alpha_spatial": 1e-3, "lambda_R": 1e-6, 
     "method": 'mua2conc', "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    
    # Config 2: Direct, no spatial basis
    {"alpha_meas": 1e4, "alpha_spatial": 1e-3, "lambda_R": 1e-6, 
     "method": 'conc', "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    
    # Config 3: Indirect with spatial basis
    {"alpha_meas": 1e4, "alpha_spatial": 1e-2, "lambda_R": 1e-6, 
     "method": 'mua2conc', "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
    
    # Config 4: Direct with spatial basis
    {"alpha_meas": 1e4, "alpha_spatial": 1e-2, "lambda_R": 1e-6, 
     "method": 'conc', "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
]
```

This results in **4 × 2 = 8** test cases (4 configs × 2 trial types).

**Tests per configuration:**

#### 2.1 Single-Subject Image Reconstruction
- **Function:** `run_pipeline_image_recon()`
- **Outputs:**
  - Reconstructed image (µM)
  - Posterior MSE (µM²)
- **References:** `single_sub_image_*.nc`, `single_sub_post_*.nc`
- **Tolerances:**
  - Image: rtol=1e-3, atol=1e-8
  - MSE: rtol=1e-3, atol=1e-12

#### 2.2 Group-Level Image Statistics
- **Function:** `run_pipeline_image_group_avg()`
- **Outputs tested:**
  - `mean`: Group average image
  - `stderr`: Standard error map
  - `tstat`: T-statistic map
  - `mse_within`: Within-subject variance
  - `mse_btw`: Between-subject variance
- **References:** `image_group_*_{direct|indirect}_*.nc`
- **Tolerances:** Same as channel-space tests

## Running Tests

### Run all tests:
```bash
cd tests
python test_hrf_estimation.py
python test_image_recon.py
```

### Run individual test sections:
Edit test files to comment out sections you want to skip.

### Inspect intermediate outputs:
```bash
# View saved HRF estimates
ls tests/data/tmp/sub-586/processed_data/

# Load intermediate results in Python
import xarray as xr
import pickle, gzip

# Load per-subject HRF
with gzip.open('tests/data/tmp/sub-586/processed_data/sub-586_task-BS_conc_hrf_estimates_ar_irls.pkl.gz', 'rb') as f:
    results = pickle.load(f)
print(results['hrf_per_subj'])
```

## Test Workflow

### 1. Initial Setup (First Run)
When run for the first time in a new environment:
```python
# Tests execute pipeline
actual_output = run_pipeline_estimate_hrf(...)

# Load reference (created manually or from previous validated run)
reference = xr.load_dataarray('reference/single_channel_single_subj_hrf.nc')

# Compare
assert_xarray_close(actual_output, reference)

# If successful, update reference (optional)
if comparison_passed:
    actual_output.to_netcdf('reference/single_channel_single_subj_hrf.nc')
```

### 2. Regression Testing (Subsequent Runs)
After code changes:
```python
# Run pipeline with new code
actual_output = run_pipeline_estimate_hrf(...)

# Compare against established reference
reference = xr.load_dataarray('reference/single_channel_single_subj_hrf.nc')

# Fail if outputs deviate
assert_xarray_close(actual_output, reference, rtol=1e-3, atol=1e-8)
# AssertionError raised if regression detected
```

### 3. Intentional Changes
When updating algorithms intentionally:
1. Run tests (expect failures)
2. Manually verify new outputs are correct
3. Update reference files:
   ```python
   # Tests automatically update reference on successful comparison
   actual_output.to_netcdf('reference/single_channel_single_subj_hrf.nc')
   ```
4. Commit updated reference files to version control

## Interpreting Test Failures

### Failure types:

#### 1. Numerical Precision Issues
```
AssertionError: Not equal to tolerance rtol=0.001, atol=1e-08
Max absolute difference: 1.23e-07
Max relative difference: 2.34e-04
```
**Possible causes:**
- Floating-point rounding differences
- Library version changes (numpy, scipy)
- Hardware differences (CPU vs GPU)

**Resolution:**
- If differences are tiny (< 1e-6), may be acceptable
- Consider loosening tolerances slightly
- Verify core algorithm hasn't changed

#### 2. Algorithmic Changes
```
AssertionError: Not equal to tolerance rtol=0.001, atol=1e-08
Max absolute difference: 0.523
Max relative difference: 0.891
```
**Possible causes:**
- Bug fix in preprocessing
- Changed GLM implementation
- Updated motion correction

**Resolution:**
- Review code changes carefully
- Validate new outputs are scientifically correct
- Update reference files if intentional improvement
- Revert if unintended regression

#### 3. Dimensional Mismatches
```
ValueError: operands could not be broadcast together with shapes (10,2,100) (10,2,99)
```
**Possible causes:**
- Changed time windowing
- Different sampling rate handling
- Modified concatenation logic

**Resolution:**
- Check pipeline configuration consistency
- Verify input data hasn't changed
- Debug shape transformations step-by-step

## Tolerance Guidelines

### Why different tolerances?

**HRF values (atol=1e-8 M, rtol=1e-3):**
- Typical HRF magnitudes: 1e-6 to 1e-4 M (µM range)
- Absolute tolerance 1e-8 allows ~1% variation at µM scale
- Relative tolerance 1e-3 allows 0.1% proportional differences

**MSE values (atol=1e-12 M², rtol=1e-3):**
- MSE is squared units, numerically smaller
- Tighter absolute tolerance for precision
- Same relative tolerance for consistency

**T-statistics (atol=1e-8, rtol=1e-4):**
- Unitless ratio (signal/noise)
- Slightly tighter relative tolerance
- More sensitive to numerical precision

### Adjusting tolerances:

If tests fail due to minor precision issues across all outputs:
```python
# In test file
assert_xarray_close(actual, reference, rtol=1e-2, atol=1e-7)  # Looser
```

**Warning:** Loosening tolerances risks masking real regressions. Always investigate root cause first.

## Continuous Integration

### Recommended CI workflow:

```yaml
# .github/workflows/regression_tests.yml
name: Regression Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install cedalion xarray pandas numpy scipy pytest
      
      - name: Run channel-space tests
        run: python tests/test_hrf_estimation.py
      
      - name: Run image-space tests
        run: python tests/test_image_recon.py
      
      - name: Upload test artifacts
        if: failure()
        uses: actions/upload-artifact@v2
        with:
          name: test-outputs
          path: tests/data/tmp/
```

## Adding New Tests

### 1. Add new test data:
```bash
# Add new subject
mkdir tests/data/sub-999
# Copy SNIRF and events files
```

### 2. Create new test case:
```python
def test_new_feature():
    """Test new preprocessing feature."""
    # Run pipeline with new feature
    output = run_pipeline_new_feature(ROOT_DIR, ...)
    
    # Define expected output
    reference = xr.load_dataarray('reference/new_feature_output.nc')
    
    # Compare
    assert_xarray_close(output, reference, rtol=1e-3, atol=1e-8)
```

### 3. Generate initial reference:
```python
# First run: create reference manually
output = run_pipeline_new_feature(ROOT_DIR, ...)
output.to_netcdf('tests/data/reference/new_feature_output.nc')
```

### 4. Document test:
Add description to this `TESTING.md` file.

## Troubleshooting

### "Reference file not found"
```
FileNotFoundError: [Errno 2] No such file or directory: 'reference/...'
```
**Solution:** Generate reference files first by running pipeline manually and saving outputs.

### "Cannot import module"
```
ModuleNotFoundError: No module named 'cedalion'
```
**Solution:** Install dependencies: `pip install cedalion xarray pandas numpy scipy`

### "CUDA out of memory" (if using GPU)
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```
**Solution:** Run on CPU or reduce batch size in pipeline configuration.

### Tests pass locally but fail in CI
**Possible causes:**
- Different random seeds
- Library version mismatches
- Insufficient memory

**Solution:**
- Pin dependency versions in `requirements.txt`
- Set random seeds explicitly
- Increase CI machine resources

## Performance Considerations

### Test execution time:
- `test_hrf_estimation.py`: ~5-10 minutes (5 subjects, 3 runs each)
- `test_image_recon.py`: ~15-30 minutes (4 configs × 5 subjects)

### Optimizations:
1. **Reduce test data size:**
   ```python
   # Use subset of time points
   amp = amp.isel(time=slice(0, 1000))  # First 1000 samples only
   ```

2. **Test fewer configurations:**
   ```python
   # Test only critical configs
   cfg_list = [cfg_list[0], cfg_list[-1]]  # First and last only
   ```

3. **Parallel execution:**
   ```bash
   # Run tests simultaneously
   python test_hrf_estimation.py &
   python test_image_recon.py &
   wait
   ```

## Contact

For questions about testing:
- **Author:** Laura Carlton (lcarlton@bu.edu)
- **Issues:** [GitHub Issues] (if repository is on GitHub)
