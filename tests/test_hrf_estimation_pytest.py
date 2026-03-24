# test_hrf_estimation.py

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
sys.path.append(PIPELINES_DIR)

from STEP1_hrf_estimation import run_pipeline_estimate_hrf
from STEP2_get_group_average_channel import run_pipeline_channel_group_avg

# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def config():
    DATA_DIR = os.path.join(TEST_DIR, "data")
    SAVE_DIR = os.path.join(DATA_DIR, "tmp")
    os.makedirs(SAVE_DIR, exist_ok=True)

    return {
        "DATA_DIR": DATA_DIR,
        "SAVE_DIR": SAVE_DIR,
        "TASK": "BS",
        "N_RUNS": 3,
        "NOISE_MODEL": "ar_irls",
        "GOLDEN_CHANNEL": "S53D28",
        "TOLERANCES": {
            'hrf': {'rtol': 1e-3, 'atol': 1e-8},
            'mse': {'rtol': 1e-3, 'atol': 1e-12},
            'tstat': {'rtol': 1e-4, 'atol': 1e-8},
        }
    }


@pytest.fixture(scope="session")
def ref_dir(config):
    return os.path.join(config["DATA_DIR"], "reference")


# ------------------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------------------

def assert_xarray_close(actual, reference, rtol, atol):
    xrt.assert_allclose(actual, reference, rtol=rtol, atol=atol)


# ------------------------------------------------------------------------------
# TEST 1: STEP1 HRF estimation
# ------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def step1_outputs(config):
    return run_pipeline_estimate_hrf(
        ROOT_DIR=config["DATA_DIR"],
        TASK=config["TASK"],
        N_RUNS=config["N_RUNS"],
        NOISE_MODEL=config["NOISE_MODEL"],
        SAVE_DIR=config["SAVE_DIR"],
        GOLDEN_CHANNEL=config["GOLDEN_CHANNEL"]
    )


def test_step1_hrf(step1_outputs, ref_dir, config):
    actual_hrf, _ = step1_outputs

    ref = xr.load_dataarray(
        os.path.join(ref_dir, "single_channel_single_subj_hrf.nc")
    ).pint.quantify("molar")

    assert_xarray_close(
        actual_hrf,
        ref,
        **config["TOLERANCES"]["hrf"]
    )


def test_step1_mse(step1_outputs, ref_dir, config):
    _, actual_mse = step1_outputs

    ref = xr.load_dataarray(
        os.path.join(ref_dir, "single_channel_single_subj_hrf_mse.nc")
    ).pint.quantify("molar**2")

    assert_xarray_close(
        actual_mse,
        ref,
        **config["TOLERANCES"]["mse"]
    )


# ------------------------------------------------------------------------------
# TEST 2: STEP2 group stats
# ------------------------------------------------------------------------------

@pytest.fixture(scope="session")
def step2_outputs(config):
    return run_pipeline_channel_group_avg(
        ROOT_DIR=config["DATA_DIR"],
        TASK=config["TASK"],
        NOISE_MODEL=config["NOISE_MODEL"],
        GOLDEN_CHANNEL=config["GOLDEN_CHANNEL"]
    )


@pytest.mark.parametrize(
    "key, units, rtol, atol",
    [
        ("mean", "molar", 1e-3, 1e-8),
        ("stderr", "molar", 1e-3, 1e-8),
        ("tstat", None, 1e-4, 1e-8),
        ("mse_within", "molar**2", 1e-3, 1e-12),
        ("mse_btw", "molar**2", 1e-3, 1e-12),
    ]
)
def test_step2_outputs(step2_outputs, ref_dir, key, units, rtol, atol):
    actual = step2_outputs[key]

    ref = xr.load_dataarray(
        os.path.join(ref_dir, f"single_channel_group_{key}.nc")
    )

    if units:
        ref = ref.pint.quantify(units)

    assert_xarray_close(actual, ref, rtol=rtol, atol=atol)