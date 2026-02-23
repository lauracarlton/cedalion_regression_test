#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Tests for Image-Space fNIRS Analysis Pipeline

Tests the following:
1. STEP3: Per-subject image reconstruction from channel-space HRF estimates
2. STEP4: Group-level weighted averaging in image space

Tests multiple reconstruction configurations:
- Direct (conc) vs Indirect (mua2conc) methods
- With and without spatial basis functions
- Different regularization parameters

Usage:
    python test_image_recon.py

Requirements:
    - STEP1 outputs must exist (run test_hrf_estimation.py first)
    - Test data in tests/data/sub-*/nirs/
    - Reference outputs in tests/data/reference/
    - Probe geometry and forward model in tests/data/probe/

Behavior:
    - Tests 4 reconstruction configurations × 2 trial types = 8 test scenarios
    - Compares outputs against reference data
    - Fails with AssertionError if outputs deviate beyond tolerance
    - Updates reference files on successful comparison

Author: Laura Carlton
"""

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

from STEP3_image_recon_on_HRF import run_pipeline_image_recon
from STEP4_get_group_average_image import run_pipeline_image_group_avg

# ==============================================================================
# Test Configuration
# ==============================================================================

# Test data location (relative to this file)
DATA_DIR = os.path.join(TEST_DIR, "data")
REF_DIR = os.path.join(DATA_DIR, "reference")

# Analysis parameters
NOISE_MODEL = "ar_irls"
TASK = "BS"
REC_STR = 'conc'
T_WIN = [5, 8]  # Time window for HRF magnitude (seconds)

# Reconstruction configurations to test
# Tests combinations of:
# - method: 'conc' (direct) vs 'mua2conc' (indirect)
# - SB: spatial basis functions (True/False)
# - alpha_spatial: spatial regularization strength
cfg_list = [
    # Config 1: Indirect, no spatial basis
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-3,
        "lambda_R": 1e-6,
        "method": 'mua2conc',
        "SB": False,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
    # Config 2: Direct, no spatial basis
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-3,
        "lambda_R": 1e-6,
        "method": 'conc',
        "SB": False,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
    # Config 3: Indirect with spatial basis
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-2,
        "lambda_R": 1e-6,
        "method": 'mua2conc',
        "SB": True,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
    # Config 4: Direct with spatial basis
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-2,
        "lambda_R": 1e-6,
        "method": 'conc',
        "SB": True,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
]

# Numerical tolerances
TOLERANCES = {
    'image': {'rtol': 1e-3, 'atol': 1e-8},
    'mse': {'rtol': 1e-3, 'atol': 1e-12},
    'tstat': {'rtol': 1e-4, 'atol': 1e-8},
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
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    
    Raises
    ------
    AssertionError
        If arrays differ beyond tolerances.
    """
    xrt.assert_allclose(actual, reference, rtol=rtol, atol=atol)


def get_config_name(cfg):
    """Generate descriptive name for configuration."""
    method_name = "direct" if cfg["method"] == "conc" else "indirect"
    sb_name = "with_SB" if cfg["SB"] else "no_SB"
    return f"{method_name}_{sb_name}_as{cfg['alpha_spatial']:.0e}"


def build_reference_path(cfg, data_type, var_name=None):
    """
    Build reference file path based on configuration.
    
    Parameters
    ----------
    cfg : dict
        Reconstruction configuration.
    data_type : str
        'single_sub_image', 'single_sub_post', or 'image_group'.
    var_name : str, optional
        Variable name for group statistics (e.g., 'mean', 'tstat').
    
    Returns
    -------
    str
        Full path to reference file.
    """
    method_name = "direct" if cfg["method"] == "conc" else "indirect"
    alpha_spatial = cfg["alpha_spatial"]
    lambda_R = cfg["lambda_R"]
    alpha_meas = cfg["alpha_meas"]
    sigma_brain = cfg["sigma_brain"]
    sigma_scalp = cfg["sigma_scalp"]
    SB = cfg["SB"]
    
    if SB:
        base = f"{data_type}_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{method_name}"
    else:
        base = f"{data_type}_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_{method_name}"
    
    if var_name:
        filename = f"{base}_{var_name}.nc"
    else:
        filename = f"{base}.nc"
    
    return os.path.join(REF_DIR, filename)

# ==============================================================================
# Main Test Loop: Test All Configurations
# ==============================================================================

print("="*80)
print(f"IMAGE RECONSTRUCTION REGRESSION TESTS")
print("="*80)
print(f"\nTesting {len(cfg_list)} reconstruction configurations...")
print(f"  Data directory: {DATA_DIR}")
print(f"  Noise model: {NOISE_MODEL}")
print(f"  Time window: {T_WIN} seconds")
print()

all_configs_passed = True
config_results = []

for config_idx, cfg in enumerate(cfg_list, 1):
    config_name = get_config_name(cfg)
    
    print("-"*80)
    print(f"Configuration {config_idx}/{len(cfg_list)}: {config_name}")
    print("-"*80)
    print(f"  Method: {cfg['method']}")
    print(f"  Spatial basis: {cfg['SB']}")
    print(f"  alpha_meas: {cfg['alpha_meas']:.0e}")
    print(f"  alpha_spatial: {cfg['alpha_spatial']:.0e}")
    print(f"  lambda_R: {cfg['lambda_R']:.0e}")
    
    try:
        # ======================================================================
        # Test 3A: Single-Subject Image Reconstruction (STEP3)
        # ======================================================================
        
        print(f"\n[Step 3A] Running image reconstruction...")
        actual_image, actual_post = run_pipeline_image_recon(
            DATA_DIR,
            cfg,
            TASK=TASK,
            NOISE_MODEL=NOISE_MODEL,
            REC_STR=REC_STR,
            T_WIN=T_WIN,
        )
        
        print(f"  Image shape: {actual_image.shape} {actual_image.dims}")
        print(f"  MSE shape: {actual_post.shape} {actual_post.dims}")
        
        # Load and compare reference data
        ref_image_path = build_reference_path(cfg, "single_sub_image_hrf_mag")
        ref_post_path = build_reference_path(cfg, "single_sub_post_hrf_mag")
        
        print(f"\n[Step 3B] Comparing to reference...")
        ref_image = xr.load_dataarray(ref_image_path).pint.quantify("molar")
        ref_post = xr.load_dataarray(ref_post_path).pint.quantify("molar**2")
        
        # Test image
        print(f"  Testing image (rtol={TOLERANCES['image']['rtol']}, atol={TOLERANCES['image']['atol']})...")
        assert_xarray_close(
            actual_image, ref_image,
            rtol=TOLERANCES['image']['rtol'],
            atol=TOLERANCES['image']['atol']
        )
        print("    ✓ Image matches reference")
        
        # Test posterior MSE
        print(f"  Testing MSE (rtol={TOLERANCES['mse']['rtol']}, atol={TOLERANCES['mse']['atol']})...")
        assert_xarray_close(
            actual_post, ref_post,
            rtol=TOLERANCES['mse']['rtol'],
            atol=TOLERANCES['mse']['atol']
        )
        print("    ✓ MSE matches reference")
        
        # Update reference files
        actual_image.pint.dequantify().to_netcdf(ref_image_path)
        actual_post.pint.dequantify().to_netcdf(ref_post_path)
        
        print("\n  ✓ STEP3 tests passed")
        
        # ======================================================================
        # Test 3C: Group-Level Image Statistics (STEP4)
        # ======================================================================
        
        print(f"\n[Step 4A] Running group averaging...")
        group_avg = run_pipeline_image_group_avg(
            ROOT_DIR=DATA_DIR,
            cfg=cfg,
            TASK=TASK,
            NOISE_MODEL=NOISE_MODEL,
        )
        
        # Test each group statistic
        print(f"\n[Step 4B] Testing {len(group_avg)} group statistics...")
        
        group_vars = {
            'mean': {'units': 'molar', 'rtol': 1e-3, 'atol': 1e-8, 'desc': 'Group mean'},
            'stderr': {'units': 'molar', 'rtol': 1e-3, 'atol': 1e-8, 'desc': 'Standard error'},
            'tstat': {'units': None, 'rtol': 1e-4, 'atol': 1e-8, 'desc': 'T-statistic'},
            'mse_within': {'units': 'molar**2', 'rtol': 1e-3, 'atol': 1e-12, 'desc': 'Within-subj MSE'},
            'mse_btw': {'units': 'molar**2', 'rtol': 1e-3, 'atol': 1e-12, 'desc': 'Between-subj MSE'},
        }
        
        for var_name, var_config in group_vars.items():
            print(f"    Testing '{var_name}' ({var_config['desc']})...")
            
            # Load reference
            ref_path = build_reference_path(cfg, "image_group", var_name)
            ref = xr.load_dataarray(ref_path)
            if var_config['units']:
                ref = ref.pint.quantify(var_config['units'])
            
            # Compare
            assert_xarray_close(
                group_avg[var_name], ref,
                rtol=var_config['rtol'],
                atol=var_config['atol']
            )
            print(f"      ✓ {var_name} matches")
            
            # Update reference
            group_avg[var_name].pint.dequantify().to_netcdf(ref_path)
        
        print(f"\n  ✓ STEP4 tests passed")
        print(f"\n✓ Configuration {config_idx} PASSED: {config_name}")
        config_results.append((config_name, "PASSED"))
        
    except AssertionError as e:
        print(f"\n✗ Configuration {config_idx} FAILED: {config_name}")
        print(f"  Error: {e}")
        config_results.append((config_name, "FAILED"))
        all_configs_passed = False
        raise

# ==============================================================================
# Summary
# ==============================================================================

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

for config_name, status in config_results:
    symbol = "✓" if status == "PASSED" else "✗"
    print(f"  {symbol} {config_name}: {status}")

if all_configs_passed:
    print("\n✓ ALL TESTS PASSED")
    print("  All reference files updated successfully")
else:
    print("\n✗ SOME TESTS FAILED")
    print("  Reference files updated only for passing tests")

print("="*80)

