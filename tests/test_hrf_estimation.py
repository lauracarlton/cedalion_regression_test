
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Tests for Channel-Space fNIRS Analysis Pipeline

Tests the following:
1. STEP1: Per-subject HRF estimation via GLM
2. STEP2: Group-level weighted averaging in channel space

Usage:
    python test_hrf_estimation.py

Requirements:
    - Test data in tests/data/sub-*/nirs/
    - Reference outputs in tests/data/reference/
    - Probe geometry in tests/data/probe/

Behavior:
    - Compares pipeline outputs against reference data
    - Fails with AssertionError if outputs deviate beyond tolerance
    - Updates reference files only if comparison succeeds (for versioning)

Author: Laura Carlton
"""
#%%
import os
import sys

import xarray as xr
import xarray.testing as xrt
import numpy as np

# Add pipelines directory to path (relative to this file)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PIPELINES_DIR = os.path.join(PROJECT_ROOT, "pipelines")
sys.path.insert(0, PIPELINES_DIR)

from STEP1_hrf_estimation import run_pipeline_estimate_hrf
from STEP2_get_group_average_channel import run_pipeline_channel_group_avg

# ==============================================================================
# Test Configuration
# ==============================================================================

# Test data location (relative to this file)
DATA_DIR = os.path.join(TEST_DIR, "data")

# Analysis parameters
GOLDEN_CHANNEL = 'S53D28'  # Single channel used for regression testing
NOISE_MODEL = "ar_irls"    # GLM noise model
TASK = "BS"                # Task identifier
N_RUNS = 3                 # Number of runs per subject

# Output directory for intermediate results
SAVE_DIR = os.path.join(DATA_DIR, 'tmp')
os.makedirs(SAVE_DIR, exist_ok=True)

# Numerical tolerances for comparisons
# - rtol: relative tolerance (proportional differences)
# - atol: absolute tolerance (fixed differences)
TOLERANCES = {
    'hrf': {'rtol': 1e-3, 'atol': 1e-8},        # HRF values in molar
    'mse': {'rtol': 1e-3, 'atol': 1e-12},       # MSE in molar^2 (smaller)
    'tstat': {'rtol': 1e-4, 'atol': 1e-8},      # T-statistics (unitless)
}

# ==============================================================================
# Helper Functions
# ==============================================================================

def assert_xarray_close(actual, reference, rtol=1e-3, atol=1e-8):
    """
    Compare two xarray.DataArray objects numerically.
    
    Parameters
    ----------
    actual : xr.DataArray
        Actual output from pipeline.
    reference : xr.DataArray
        Expected reference output.
    rtol : float, optional
        Relative tolerance (default: 1e-3).
    atol : float, optional
        Absolute tolerance (default: 1e-8).
    
    Raises
    ------
    AssertionError
        If arrays differ beyond tolerances.
    """
    xrt.assert_allclose(actual, reference, rtol=rtol, atol=atol)

# ==============================================================================
# Test 1: Single-Subject HRF Estimation (STEP1)
# ==============================================================================

print("="*80)
print("TEST 1: Single-Subject HRF Estimation")
print("="*80)

print(f"\nRunning STEP1 pipeline...")
print(f"  Data directory: {DATA_DIR}")
print(f"  Golden channel: {GOLDEN_CHANNEL}")
print(f"  Noise model: {NOISE_MODEL}")
print(f"  Number of runs: {N_RUNS}")

actual_hrf, actual_mse = run_pipeline_estimate_hrf(
    ROOT_DIR=DATA_DIR,
    TASK=TASK,
    N_RUNS=N_RUNS,
    NOISE_MODEL=NOISE_MODEL,
    SAVE_DIR=SAVE_DIR,
    GOLDEN_CHANNEL=GOLDEN_CHANNEL
)

print(f"\nPipeline completed. Output shapes:")
print(f"  HRF: {actual_hrf.shape} {actual_hrf.dims}")
print(f"  MSE: {actual_mse.shape} {actual_mse.dims}")

# Load reference data
ref_dir = os.path.join(DATA_DIR, "reference")
ref_hrf_path = os.path.join(ref_dir, "single_channel_single_subj_hrf.nc")
ref_mse_path = os.path.join(ref_dir, "single_channel_single_subj_hrf_mse.nc")

print(f"\nLoading reference data...")
ref_hrf = xr.load_dataarray(ref_hrf_path).pint.quantify("molar")
ref_mse = xr.load_dataarray(ref_mse_path).pint.quantify("molar**2")

# Compare outputs
print(f"\nComparing outputs to reference...")
try:
    # Test HRF estimates
    print(f"  Testing HRF (rtol={TOLERANCES['hrf']['rtol']}, atol={TOLERANCES['hrf']['atol']})...")
    assert_xarray_close(
        actual_hrf, ref_hrf, 
        rtol=TOLERANCES['hrf']['rtol'], 
        atol=TOLERANCES['hrf']['atol']
    )
    print("    ✓ HRF matches reference")
    
    # Test MSE estimates
    print(f"  Testing MSE (rtol={TOLERANCES['mse']['rtol']}, atol={TOLERANCES['mse']['atol']})...")
    assert_xarray_close(
        actual_mse, ref_mse,
        rtol=TOLERANCES['mse']['rtol'],
        atol=TOLERANCES['mse']['atol']
    )
    print("    ✓ MSE matches reference")

    print("\n✓ TEST 1 PASSED: All outputs match reference")

    # Update reference files (maintains version consistency)
    print("\nUpdating reference files for version control...")
    actual_hrf.pint.dequantify().to_netcdf(ref_hrf_path)
    actual_mse.pint.dequantify().to_netcdf(ref_mse_path)
    print("  Reference files updated successfully")

except AssertionError as e:
    print("\n✗ TEST 1 FAILED: Output does not match reference")
    print("  Reference files NOT updated")
    print(f"\nError details:\n{e}")
    raise

# ==============================================================================
# Test 2: Group-Level Statistics (STEP2)
# ==============================================================================

print("\n" + "="*80)
print("TEST 2: Group-Level Channel-Space Statistics")
print("="*80)

print(f"\nRunning STEP2 pipeline...")
group_avg = run_pipeline_channel_group_avg(
    ROOT_DIR=DATA_DIR,
    TASK=TASK,
    NOISE_MODEL=NOISE_MODEL,
    GOLDEN_CHANNEL=GOLDEN_CHANNEL
)

print(f"\nPipeline completed. Testing {len(group_avg)} output variables...")

# Define variables to test with their units and tolerances
TESTING_VARS = {
    'mean': {'units': 'molar', 'rtol': 1e-3, 'atol': 1e-8, 'desc': 'Weighted mean HRF'},
    'stderr': {'units': 'molar', 'rtol': 1e-3, 'atol': 1e-8, 'desc': 'Standard error'},
    'tstat': {'units': None, 'rtol': 1e-4, 'atol': 1e-8, 'desc': 'T-statistic'},
    'mse_within': {'units': 'molar**2', 'rtol': 1e-3, 'atol': 1e-12, 'desc': 'Within-subject MSE'},
    'mse_btw': {'units': 'molar**2', 'rtol': 1e-3, 'atol': 1e-12, 'desc': 'Between-subject MSE'},
}

all_passed = True
for key, config in TESTING_VARS.items():
    print(f"\n  Testing '{key}' ({config['desc']})...")
    
    # Load reference
    ref_path = os.path.join(ref_dir, f"single_channel_group_{key}.nc")
    ref = xr.load_dataarray(ref_path)
    if config['units']:
        ref = ref.pint.quantify(config['units'])
    
    # Compare
    try:
        assert_xarray_close(
            group_avg[key], ref,
            rtol=config['rtol'],
            atol=config['atol']
        )
        print(f"    ✓ {key} matches reference (rtol={config['rtol']}, atol={config['atol']})")
        
        # Update reference file
        group_avg[key].pint.dequantify().to_netcdf(ref_path)
        
    except AssertionError as e:
        print(f"    ✗ {key} does NOT match reference")
        print(f"      Error: {e}")
        all_passed = False
        raise

if all_passed:
    print("\n✓ TEST 2 PASSED: All group statistics match reference")
    print("  Reference files updated successfully")
else:
    print("\n✗ TEST 2 FAILED: Some outputs do not match reference")

print("\n" + "="*80)
print("ALL TESTS COMPLETED SUCCESSFULLY")
print("="*80)

# %%
