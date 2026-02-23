
#%%
import os
import sys

import xarray as xr
import xarray.testing as xrt

import numpy as np

sys.path.append( os.path.join("/projectnb", "nphfnirs", "s", "users", "lcarlton", "ANALYSIS_CODE", "cedalion_regression_test", "pipelines"))
from STEP1_hrf_estimation import run_pipeline_estimate_hrf
from STEP2_get_group_average_channel import run_pipeline_channel_group_avg


def assert_xarray_close(actual, reference, rtol=1e-3, atol=1e-8):
    """
    Compare two xarray.DataArray objects numerically.
    """

    # Checks dims, coords, values, and metadata
    xrt.assert_allclose(actual, reference, rtol=rtol, atol=atol)
    
GOLDEN_CHANNEL = 'S53D28'

ROOT_DIR = os.path.join("/projectnb", "nphfnirs", "s", "users", "lcarlton", "ANALYSIS_CODE", "cedalion_regression_test", "tests", "data")
NOISE_MODEL = "ar_irls"
TASK = "BS" 
N_RUNS = 3
SAVE_DIR = os.path.join(ROOT_DIR, 'tmp')
os.makedirs(SAVE_DIR, exist_ok=True)

#%% STEP 1 
actual_hrf, actual_mse = run_pipeline_estimate_hrf(
                                                    ROOT_DIR=ROOT_DIR, 
                                                    TASK=TASK, 
                                                    N_RUNS=N_RUNS,
                                                    NOISE_MODEL=NOISE_MODEL,
                                                    SAVE_DIR=SAVE_DIR,
                                                    GOLDEN_CHANNEL=GOLDEN_CHANNEL
                                            )


# load in the reference 
ref_hrf_path = os.path.join(
                                ROOT_DIR, "reference/single_channel_single_subj_hrf.nc"
                            )

ref_mse_path = os.path.join(
                                ROOT_DIR, "reference/single_channel_single_subj_hrf_mse.nc"
                            )

ref_hrf = xr.load_dataarray(ref_hrf_path).pint.quantify("molar")
ref_mse = xr.load_dataarray(ref_mse_path).pint.quantify("molar**2")

# run comparison 
try:
    assert_xarray_close(actual_hrf, ref_hrf, rtol=1e-3, atol=1e-8)
    assert_xarray_close(actual_mse, ref_mse, rtol=1e-3, atol=1e-12)

    print("Outputs match reference.")

    # Only overwrite if comparison succeeded
    print("Overwriting reference files...")

    actual_hrf.pint.dequantify().to_netcdf(ref_hrf_path)
    actual_mse.pint.dequantify().to_netcdf(ref_mse_path)

    print("Reference updated successfully.")

except AssertionError as e:
    print("Comparison failed. Reference NOT updated.")
    raise e

#%% STEP 2 
group_avg = run_pipeline_channel_group_avg( 
                                            ROOT_DIR=ROOT_DIR, 
                                            TASK=TASK, 
                                            NOISE_MODEL=NOISE_MODEL,
                                            GOLDEN_CHANNEL=GOLDEN_CHANNEL
                                            )

TESTING_VARS = ['stderr', 'tstat', 'mse_btw', 'mse_within', 'mean']

for key in TESTING_VARS: 

    print(f'Testing {key}')
    # load in the reference 
    ref = xr.load_dataarray( ROOT_DIR + f"/reference/single_channel_group_{key}.nc")
    if key == 'stderr' or key == 'mean':
        ref = ref.pint.quantify('molar')
        rtol = 1e-3
        atol = 1e-8
    elif key == 'mse_btw' or key == 'mse_within':
        ref = ref.pint.quantify('molar**2')
        rtol = 1e-3
        atol = 1e-12
    else: 
        rtol = 1e-4
        atol = 1e-8

    try:
        assert_xarray_close(group_avg[key], ref, rtol=rtol, atol=atol)
        print(f"\tOutputs for {key} match reference.")

        # Only overwrite if comparison succeeded
        print("\tOverwriting reference files...")

        group_avg[key].pint.dequantify().to_netcdf(ROOT_DIR + f"/reference/single_channel_group_{key}.nc")
        print("\tReference updated successfully.")

    except AssertionError as e:
        print("\tComparison failed. Reference NOT updated.")
        raise e

# %%
