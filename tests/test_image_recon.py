#%%
import os
import sys

import xarray as xr
import xarray.testing as xrt

import numpy as np

sys.path.append( os.path.join("/projectnb", "nphfnirs", "s", "users", "lcarlton", "ANALYSIS_CODE", "cedalion_regression_test", "pipelines"))
from STEP3_image_recon_on_HRF import run_pipeline_image_recon
from STEP4_get_group_average_image import run_pipeline_image_group_avg


def assert_xarray_close(actual, reference, rtol=1e-3, atol=1e-8):
    """
    Compare two xarray.DataArray objects numerically.
    """

    # Checks dims, coords, values, and metadata
    xrt.assert_allclose(actual, reference, rtol=rtol, atol=atol)
    


ROOT_DIR = os.path.join("/projectnb", "nphfnirs", "s", "users", "lcarlton", "ANALYSIS_CODE", "cedalion_regression_test", "tests", "data")
NOISE_MODEL = "ar_irls"
TASK = "BS" 
REC_STR = 'conc'
T_WIN = [5,8]


cfg_list = [
    {"alpha_meas": 1e4, "alpha_spatial": 1e-3, "lambda_R": 1e-6, "method": 'mua2conc', "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-3, "lambda_R": 1e-6, "method": 'conc', "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-2, "lambda_R": 1e-6, "method": 'mua2conc', "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-2, "lambda_R": 1e-6, "method": 'conc', "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
]


# %%
for cfg in cfg_list:
    actual_image, actual_post = run_pipeline_image_recon(ROOT_DIR, 
                                                            cfg, 
                                                            TASK=TASK, 
                                                            NOISE_MODEL=NOISE_MODEL, 
                                                            REC_STR=REC_STR, 
                                                            T_WIN=T_WIN,
                                                    )
    
    # load in the reference 
    method = cfg["method"]
    if method == 'conc':
        direct_name = "direct"
    else:
        direct_name = "indirect"

    SB = cfg["SB"]
    sigma_brain = cfg["sigma_brain"]
    sigma_scalp = cfg["sigma_scalp"]
    alpha_meas = cfg["alpha_meas"]
    alpha_spatial = cfg["alpha_spatial"]
    lambda_R = cfg["lambda_R"]

    if SB:
        ref_image_path = os.path.join(
            ROOT_DIR, 
            'reference',
            f"single_sub_image_hrf_mag_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}.nc",
        )
        ref_post_path = os.path.join(
            ROOT_DIR, 
            'reference',
            f"single_sub_post_hrf_mag_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}.nc",
        )
    else:
        ref_image_path = os.path.join(
            ROOT_DIR,
            'reference',
            f"single_sub_image_hrf_mag_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_{direct_name}.nc",
        )

        ref_post_path = os.path.join(
            ROOT_DIR,
            'reference',
            f"single_sub_post_hrf_mag_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_{direct_name}.nc",
        )


    ref_image = xr.load_dataarray(ref_image_path).pint.quantify("molar")
    ref_post = xr.load_dataarray(ref_post_path).pint.quantify("molar**2")

    # run comparison 
    try:
        assert_xarray_close(actual_image, ref_image, rtol=1e-3, atol=1e-8)
        assert_xarray_close(actual_post, ref_post, rtol=1e-3, atol=1e-12)

        print("Outputs match reference.")

        # Only overwrite if comparison succeeded
        print("Overwriting reference files...")

        actual_image.pint.dequantify().to_netcdf(ref_image_path)
        actual_post.pint.dequantify().to_netcdf(ref_post_path)

        print("Reference updated successfully.")

    except AssertionError as e:
        print("Comparison failed. Reference NOT updated.")
        raise e


    group_avg = run_pipeline_image_group_avg( 
                                                ROOT_DIR=ROOT_DIR, 
                                                cfg=cfg,
                                                TASK=TASK, 
                                                NOISE_MODEL=NOISE_MODEL,
                                                )

    TESTING_VARS = ['stderr', 'tstat', 'mse_btw', 'mse_within', 'mean']
    for key in TESTING_VARS: 

        print(f'Testing {key}')
        # load in the reference 
        ref = xr.load_dataarray( ROOT_DIR + f"/reference/image_group_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{key}.nc")
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

            group_avg[key].pint.dequantify().to_netcdf(ROOT_DIR + f"/reference/image_group_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{key}.nc")
            print("\tReference updated successfully.")

        except AssertionError as e:
            print("\tComparison failed. Reference NOT updated.")
            raise e

# %%
