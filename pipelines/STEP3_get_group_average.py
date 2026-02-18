"""
FIG7_STEP3_get_group_average.py

Collect per-subject image-space reconstructions and compute group-level
weighted averages and statistics. The script aggregates per-subject
pickles produced by the image-reconstruction step, computes within- and
between-subject variances, and writes a group-summary pickle per
reconstruction configuration.

Usage
-----
Run after all per-subject image reconstructions have completed::

        python FIG8_STEP3_get_group_average.py

Configurables (near top of file)
--------------------------------
- ROOT_DIR: dataset root containing BIDS-like subject folders.
- SAVE_DIR: directory where per-subject image results are stored.
- PROBE_DIR: forward-model directory used to load Adot and G-matrices.
- TASK: task name used in per-subject filenames.
- NOISE_MODEL, fname_flag: filename conventions used to find files.
- FNAME_FLAG: filename convention for output files - either 'ts' or 'mag'.
- EXCLUDED: list of subject IDs to skip (e.g. problematic subjects).
- TRIAL_TYPES: list of trial types to process.

Outputs
-------
- Per-configuration gzipped pickle under
    <ROOT_DIR>/derivatives/cedalion/processed_data/image_space/ containing
    'X_hrf_ts', 'X_mse', 'X_tstat', 'X_std_err', and related group summaries.

Dependencies
------------
- cedalion, xarray, numpy and project helper modules (image_recon_func,
    processing_func, spatial_basis_funs) loaded from the modules/ directory.

Author: Laura Carlton
"""

# %%
import os
import pickle
import gzip
import sys

import xarray as xr
import numpy as np

from cedalion import units
from cedalion.io.forward_model import load_Adot

sys.path.append("/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/")
import image_recon_func as irf  # noqa: E402
import processing_func as pf  # noqa: E402
import spatial_basis_func as sbf  # noqa: E402

ROOT_DIR = os.path.join("/", "projectnb", "nphfnirs", "s", "datasets", "BSMW_Laura_Miray_2025", "BS")
TASK = "BS"
NOISE_MODEL = "ar_irls"
FNAME_FLAG = "mag"
lambda_R = 1e-6

# Whether per-measurement C_meas was used when performing image reconstructions
# Some filename templates expect a Cmeas_name variable; define the flag and
# derived name here to avoid NameError in templates.

CMEAS_FLAG = True
if CMEAS_FLAG:
    Cmeas_name = "Cmeas"
else:
    Cmeas_name = "noCmeas"

EXCLUDED = ["sub-538", "sub-549", "sub-547"]
TRIAL_TYPES = ["right", "left"]
optional_flag = ''

SAVE_DIR = os.path.join(ROOT_DIR, "derivatives", "cedalion", "processed_data", "image_space")
PROBE_DIR = os.path.join(ROOT_DIR, "derivatives", "cedalion", "fw", "probe")

dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if "sub" in d and d not in EXCLUDED]

# %%

cfg_list = [
    {"alpha_meas": 1e4, "alpha_spatial": 1e-3, "lambda_R": lambda_R, "DIRECT": False, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-3, "lambda_R": lambda_R, "DIRECT": True, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-2, "lambda_R": lambda_R, "DIRECT": False, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e4, "alpha_spatial": 1e-2, "lambda_R": lambda_R, "DIRECT": True, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
   ]


for cfg in cfg_list:

    DIRECT = cfg["DIRECT"]
    SB = cfg["SB"]
    sigma_brain = cfg["sigma_brain"]
    sigma_scalp = cfg["sigma_scalp"]
    alpha_meas = cfg["alpha_meas"]
    alpha_spatial = cfg["alpha_spatial"]
    lambda_R = cfg["lambda_R"]

    if DIRECT:
        direct_name = "direct"
    else:
        direct_name = "indirect"

    all_trial_all_subj_X_hrf_ts = None
    print(f"alpha_meas = {alpha_meas}, alpha_spatial = {alpha_spatial}, lambda_spatial_depth = {lambda_R}, SB = {SB}, {direct_name}")

    for trial_type in TRIAL_TYPES:
        all_subj_X_hrf_ts = []
        all_subj_X_mse = []
        print(f"\ttrial_type - {trial_type}")

        for subj in subject_list:
            folderpath = os.path.join(SAVE_DIR, subj)

            if SB:
                filepath = os.path.join(
                    SAVE_DIR,
                    subj,
                    f"{subj}_task-{TASK}_image_hrf_{FNAME_FLAG}_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{Cmeas_name}_{NOISE_MODEL}{optional_flag}.pkl.gz",
                )
            else:
                filepath = os.path.join(
                    SAVE_DIR,
                    subj,
                    f"{subj}_task-{TASK}_image_hrf_{FNAME_FLAG}_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_{direct_name}_{Cmeas_name}_{NOISE_MODEL}{optional_flag}.pkl.gz",
                )

            with gzip.open(filepath, "rb") as f:
                results = pickle.load(f)

            X_hrf_ts = results["X_hrf"].sel(trial_type=trial_type) #.drop_vars('trial_type')
            X_mse = results["X_mse"].sel(trial_type=trial_type)#.drop_vars('trial_type')
            all_subj_X_hrf_ts.append(X_hrf_ts)
            all_subj_X_mse.append(X_mse)

        all_subj_X_hrf_ts_xr = xr.concat(all_subj_X_hrf_ts, dim="subj")
        all_subj_X_mse_xr = xr.concat(all_subj_X_mse, dim="subj")

        X_hrf_ts_mean = all_subj_X_hrf_ts_xr.mean("subj", skipna=True)

        all_subj_X_hrf_ts_tmp = all_subj_X_hrf_ts_xr.where(~np.isnan(all_subj_X_hrf_ts_xr), drop=True)
        all_subj_X_mse_tmp = all_subj_X_mse_xr.where(~np.isnan(all_subj_X_mse_xr), drop=True)

        X_hrf_ts_mean_weighted = (all_subj_X_hrf_ts_tmp / all_subj_X_mse_tmp).sum("subj") / (1 / all_subj_X_mse_xr).sum(
            "subj"
        )

        X_mse_mean_within_subject = 1 / (1 / all_subj_X_mse_tmp).sum("subj")
        X_mse_mean_within_subject = X_mse_mean_within_subject.assign_coords({"trial_type": trial_type})

        X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_ts_tmp - X_hrf_ts_mean_weighted) ** 2
        X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp / all_subj_X_mse_tmp

        X_mse_weighted_between_subjects = (
            X_mse_weighted_between_subjects.mean("subj") * X_mse_mean_within_subject
        )  # normalized by the within subject variances as weights

        X_mse_weighted_between_subjects = X_mse_weighted_between_subjects.pint.dequantify()

        X_mse_btw_within_sum_subj = all_subj_X_mse_tmp.pint.dequantify() + X_mse_weighted_between_subjects.pint.dequantify()
        denom = (1 / X_mse_btw_within_sum_subj).sum("subj")

        X_hrf_ts_mean_weighted = (all_subj_X_hrf_ts_tmp / X_mse_btw_within_sum_subj).sum("subj")
        X_hrf_ts_mean_weighted = X_hrf_ts_mean_weighted / denom

        mse_total = 1 / denom

        X_stderr_weighted = np.sqrt(mse_total)
        X_tstat = X_hrf_ts_mean_weighted / X_stderr_weighted

        
        if all_trial_all_subj_X_hrf_ts is None:

            all_trial_all_subj_X_hrf_ts = all_subj_X_hrf_ts_xr
            all_trial_all_subj_X_mse = all_subj_X_mse_xr

            all_trial_X_hrf_ts = X_hrf_ts_mean
            all_trial_X_hrf_ts_weighted = X_hrf_ts_mean_weighted
            all_trial_X_stderr = X_stderr_weighted
            all_trial_X_tstat = X_tstat
            all_trial_X_mse_between = X_mse_weighted_between_subjects
            all_trial_X_mse_within = X_mse_mean_within_subject
        else:

            all_trial_all_subj_X_hrf_ts = xr.concat([all_trial_all_subj_X_hrf_ts, all_subj_X_hrf_ts_xr], dim="trial_type")
            all_trial_all_subj_X_mse = xr.concat([all_trial_all_subj_X_mse, all_subj_X_mse_xr], dim="trial_type")

            all_trial_X_hrf_ts = xr.concat([all_trial_X_hrf_ts, X_hrf_ts_mean], dim="trial_type")
            all_trial_X_hrf_ts_weighted = xr.concat(
                [all_trial_X_hrf_ts_weighted, X_hrf_ts_mean_weighted], dim="trial_type"
            )
            all_trial_X_stderr = xr.concat([all_trial_X_stderr, X_stderr_weighted], dim="trial_type")
            all_trial_X_tstat = xr.concat([all_trial_X_tstat, X_tstat], dim="trial_type")
            all_trial_X_mse_between = xr.concat(
                [all_trial_X_mse_between, X_mse_weighted_between_subjects], dim="trial_type"
            )
            all_trial_X_mse_within = xr.concat([all_trial_X_mse_within, X_mse_mean_within_subject], dim="trial_type")

    results = {
        "X_hrf_ts": all_trial_all_subj_X_hrf_ts,
        "X_mse": all_trial_all_subj_X_mse,
        "X_std_err": all_trial_X_stderr,
        "X_tstat": all_trial_X_tstat,
        "X_mse_between": all_trial_X_mse_between,
        "X_hrf_ts_mean": all_trial_X_hrf_ts,
        "X_hrf_ts_weighted": all_trial_X_hrf_ts_weighted,
        "X_mse_within": all_trial_X_mse_within,
    }

    if SB:
        filepath = os.path.join(
            SAVE_DIR,
            f"task-{TASK}_image_hrf_{FNAME_FLAG}_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_Cmeas_{NOISE_MODEL}{optional_flag}.pkl.gz",
        )
    else:
        filepath = os.path.join(
            SAVE_DIR,
            f"task-{TASK}_image_hrf_{FNAME_FLAG}_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_{direct_name}_Cmeas_{NOISE_MODEL}{optional_flag}.pkl.gz",
        )

    print(f"Saving to {filepath}")
    file = gzip.GzipFile(filepath, "wb")
    file.write(pickle.dumps(results))
    file.close()

# %%
