# test_image_recon.py

import os
import sys
import pytest
import xarray as xr
import xarray.testing as xrt

# ------------------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------------------

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PIPELINES_DIR = os.path.join(PROJECT_ROOT, "pipelines")
sys.path.insert(0, PIPELINES_DIR)

from STEP3_image_recon_on_HRF import run_pipeline_image_recon
from STEP4_get_group_average_image import run_pipeline_image_group_avg

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config():
    DATA_DIR = os.path.join(TEST_DIR, "data")
    REF_DIR = os.path.join(DATA_DIR, "reference")

    return {
        "DATA_DIR": DATA_DIR,
        "REF_DIR": REF_DIR,
        "TASK": "BS",
        "NOISE_MODEL": "ar_irls",
        "REC_STR": "conc",
        "T_WIN": [5, 8],
    }


@pytest.fixture(scope="session")
def cfg_list():
    return [
        {
            "alpha_meas": 1e4,
            "alpha_spatial": 1e-3,
            "lambda_R": 1e-6,
            "method": 'mua2conc',
            "SB": False,
            "sigma_brain": 1,
            "sigma_scalp": 5
        },
        {
            "alpha_meas": 1e4,
            "alpha_spatial": 1e-3,
            "lambda_R": 1e-6,
            "method": 'conc',
            "SB": False,
            "sigma_brain": 1,
            "sigma_scalp": 5
        },
        {
            "alpha_meas": 1e4,
            "alpha_spatial": 1e-2,
            "lambda_R": 1e-6,
            "method": 'mua2conc',
            "SB": True,
            "sigma_brain": 1,
            "sigma_scalp": 5
        },
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


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

def assert_xarray_close(actual, reference, rtol, atol):
    xrt.assert_allclose(actual, reference, rtol=rtol, atol=atol)


def build_reference_path(ref_dir, cfg, data_type, var_name=None):
    method_name = "direct" if cfg["method"] == "conc" else "indirect"

    if cfg["SB"]:
        base = (
            f"{data_type}_as-{cfg['alpha_spatial']:.0e}"
            f"_ls-{cfg['lambda_R']:.0e}"
            f"_am-{cfg['alpha_meas']:.0e}"
            f"_sb-{cfg['sigma_brain']}"
            f"_ss-{cfg['sigma_scalp']}_{method_name}"
        )
    else:
        base = (
            f"{data_type}_as-{cfg['alpha_spatial']:.0e}"
            f"_ls-{cfg['lambda_R']:.0e}"
            f"_am-{cfg['alpha_meas']:.0e}_{method_name}"
        )

    if var_name:
        base += f"_{var_name}"

    return os.path.join(ref_dir, f"{base}.nc")


# ------------------------------------------------------------------------------
# STEP3 fixtures (run once per config)
# ------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def step3_outputs(config, cfg_list):
    outputs = {}
    for cfg in cfg_list:
        key = str(cfg)
        outputs[key] = run_pipeline_image_recon(
            config["DATA_DIR"],
            cfg,
            TASK=config["TASK"],
            NOISE_MODEL=config["NOISE_MODEL"],
            REC_STR=config["REC_STR"],
            T_WIN=config["T_WIN"],
        )
    return outputs


@pytest.fixture(scope="session")
def step4_outputs(config, cfg_list):
    outputs = {}
    for cfg in cfg_list:
        key = str(cfg)
        outputs[key] = run_pipeline_image_group_avg(
            ROOT_DIR=config["DATA_DIR"],
            cfg=cfg,
            TASK=config["TASK"],
            NOISE_MODEL=config["NOISE_MODEL"],
        )
    return outputs


# ------------------------------------------------------------------------------
# TEST STEP3: Image + posterior
# ------------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-3,
        "lambda_R": 1e-6,
        "method": 'mua2conc',
        "SB": False,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-3,
        "lambda_R": 1e-6,
        "method": 'conc',
        "SB": False,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-2,
        "lambda_R": 1e-6,
        "method": 'mua2conc',
        "SB": True,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-2,
        "lambda_R": 1e-6,
        "method": 'conc',
        "SB": True,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
])
def test_step3_image(cfg, config):
    actual_image, actual_post = run_pipeline_image_recon(
        config["DATA_DIR"],
        cfg,
        TASK=config["TASK"],
        NOISE_MODEL=config["NOISE_MODEL"],
        REC_STR=config["REC_STR"],
        T_WIN=config["T_WIN"],
    )

    ref_image = xr.load_dataarray(
        build_reference_path(config["REF_DIR"], cfg, "single_sub_image_hrf_mag")
    ).pint.quantify("molar")

    ref_post = xr.load_dataarray(
        build_reference_path(config["REF_DIR"], cfg, "single_sub_post_hrf_mag")
    ).pint.quantify("molar**2")

    assert_xarray_close(actual_image, ref_image, rtol=1e-3, atol=1e-8)
    assert_xarray_close(actual_post, ref_post, rtol=1e-3, atol=1e-12)


# ------------------------------------------------------------------------------
# TEST STEP4: Group stats
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "var_name, units, rtol, atol",
    [
        ("mean", "molar", 1e-3, 1e-8),
        ("stderr", "molar", 1e-3, 1e-8),
        ("tstat", None, 1e-4, 1e-8),
        ("mse_within", "molar**2", 1e-3, 1e-12),
        ("mse_btw", "molar**2", 1e-3, 1e-12),
    ]
)
@pytest.mark.parametrize("cfg", [
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-3,
        "lambda_R": 1e-6,
        "method": 'mua2conc',
        "SB": False,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-3,
        "lambda_R": 1e-6,
        "method": 'conc',
        "SB": False,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-2,
        "lambda_R": 1e-6,
        "method": 'mua2conc',
        "SB": True,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
    {
        "alpha_meas": 1e4,
        "alpha_spatial": 1e-2,
        "lambda_R": 1e-6,
        "method": 'conc',
        "SB": True,
        "sigma_brain": 1,
        "sigma_scalp": 5
    },
])
def test_step4_group(cfg, var_name, units, rtol, atol, config):
    group_avg = run_pipeline_image_group_avg(
        ROOT_DIR=config["DATA_DIR"],
        cfg=cfg,
        TASK=config["TASK"],
        NOISE_MODEL=config["NOISE_MODEL"],
    )

    ref = xr.load_dataarray(
        build_reference_path(config["REF_DIR"], cfg, "image_group", var_name)
    )

    if units:
        ref = ref.pint.quantify(units)

    assert_xarray_close(group_avg[var_name], ref, rtol=rtol, atol=atol)