"""
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


# Whether per-measurement C_meas was used when performing image reconstructions
# Some filename templates expect a Cmeas_name variable; define the flag and
# derived name here to avoid NameError in templates.


def run_pipeline_image_group_avg(
                                    ROOT_DIR, 
                                    cfg,
                                    TASK='BS', 
                                    NOISE_MODEL='ar_irls',
                                    TRIAL_TYPES=['right', 'left']
                                    ):
    """
    Compute weighted group-level statistics for image-space reconstructions.
    
    Aggregates per-subject image reconstructions from STEP3 across subjects using
    inverse-variance weighting. Produces group-level activation maps with
    within-subject and between-subject variance estimates for statistical inference.
    
    Parameters
    ----------
    ROOT_DIR : str
        Root directory containing per-subject image reconstructions from STEP3.
        Expected: ROOT_DIR/tmp/sub-XXX/processed_data/*_image_*.pkl.gz
    cfg : dict
        Reconstruction configuration matching STEP3 (used for filename matching):
        - 'method' : str, 'conc' or 'mua2conc'
        - 'alpha_meas' : float
        - 'alpha_spatial' : float
        - 'lambda_R' : float
        - 'SB' : bool
        - 'sigma_brain' : float
        - 'sigma_scalp' : float
    TASK : str, optional
        Task identifier (default: 'BS').
    NOISE_MODEL : str, optional
        Noise model label: 'ols' or 'ar_irls' (default: 'ar_irls').
    TRIAL_TYPES : list of str, optional
        Trial types to aggregate (default: ['right', 'left']).
    
    Returns
    -------
    results : dict
        Dictionary containing group statistics for all vertices/flatmap positions:
        - 'mean' : Weighted mean image (xr.DataArray, units: molar)
        - 'stderr' : Standard error (xr.DataArray, units: molar)
        - 'tstat' : T-statistic (xr.DataArray, unitless)
        - 'mse_within' : Within-subject MSE (xr.DataArray, units: molar^2)
        - 'mse_btw' : Between-subject MSE (xr.DataArray, units: molar^2)
        
        All arrays have dimensions (trial_type, chromo, flatmap/vertex).
    
    Notes
    -----
    Weighting scheme (same as STEP2):
    - Each subject weighted by 1 / (MSE_within + MSE_between)
    - Accounts for both measurement noise and inter-subject variability
    
    Statistical formulation:
    - Within-subject variance: 1 / Σ(1/σ²_i) per vertex
    - Between-subject variance: Weighted squared deviations from group mean
    - Standard error: √(1 / Σ(1 / (σ²_within + σ²_between)))
    - T-statistic: mean / SE at each spatial location
    
    Output interpretation:
    - 'mean': Average activation magnitude across subjects
    - 'stderr': Uncertainty in group estimate (accounts for sample size)
    - 'tstat': Standardized effect size (for thresholding/visualization)
    - 'mse_btw': Inter-subject variability (anatomical/functional differences)
    - 'mse_within': Measurement noise (sensor quality, SNR)
    
    Example
    -------
    >>> cfg = {"method": "conc", "alpha_meas": 1e4, "alpha_spatial": 1e-3,
    ...        "lambda_R": 1e-6, "SB": True, "sigma_brain": 1, "sigma_scalp": 5}
    >>> results = run_pipeline_image_group_avg(
    ...     ROOT_DIR="/data/BS_bids",
    ...     cfg=cfg,
    ...     TASK="BS",
    ...     TRIAL_TYPES=['right', 'left']
    ... )
    >>> # Plot t-statistic map for HbO, right hand condition
    >>> tstat = results['tstat'].sel(chromo='HbO', trial_type='right')
    """

    PROBE_DIR = os.path.join(ROOT_DIR, 'probe')

    dirs = os.listdir(ROOT_DIR)
    subject_list = [d for d in dirs if "sub" in d]

    method = cfg["method"]
    SB = cfg["SB"]
    sigma_brain = cfg["sigma_brain"]
    sigma_scalp = cfg["sigma_scalp"]
    alpha_meas = cfg["alpha_meas"]
    alpha_spatial = cfg["alpha_spatial"]
    lambda_R = cfg["lambda_R"]

    if method == 'conc':
        direct_name = "direct"
    else:
        direct_name = "indirect"

    all_trial_X_hrf_ts = None
    print(f"alpha_meas = {alpha_meas}, alpha_spatial = {alpha_spatial}, lambda_spatial_depth = {lambda_R}, SB = {SB}, {direct_name}")

    for trial_type in TRIAL_TYPES:
        all_subj_X_hrf_ts = []
        all_subj_X_mse = []
        print(f"\ttrial_type - {trial_type}")

        for subj in subject_list:
            folderpath = os.path.join(ROOT_DIR, 'tmp', subj, 'processed_data')

            if SB:
                filepath = os.path.join(
                    folderpath,
                    f"{subj}_task-{TASK}_image_hrf_mag_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{NOISE_MODEL}.pkl.gz",
                )
            else:
                filepath = os.path.join(
                   folderpath,
                    f"{subj}_task-{TASK}_image_hrf_mag_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_{direct_name}_{NOISE_MODEL}.pkl.gz",
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

        X_mse_btw_within_sum_subj = all_subj_X_mse_tmp + X_mse_weighted_between_subjects
        denom = (1 / X_mse_btw_within_sum_subj).sum("subj")

        X_hrf_ts_mean_weighted = (all_subj_X_hrf_ts_tmp / X_mse_btw_within_sum_subj).sum("subj")
        X_hrf_ts_mean_weighted = X_hrf_ts_mean_weighted / denom

        mse_total = 1 / denom

        X_stderr_weighted = np.sqrt(mse_total)
        X_tstat = X_hrf_ts_mean_weighted / X_stderr_weighted

        
        if all_trial_X_hrf_ts is None:
            all_trial_X_hrf_ts = X_hrf_ts_mean
            all_trial_X_hrf_ts_weighted = X_hrf_ts_mean_weighted
            all_trial_X_stderr = X_stderr_weighted
            all_trial_X_tstat = X_tstat
            all_trial_X_mse_between = X_mse_weighted_between_subjects
            all_trial_X_mse_within = X_mse_mean_within_subject
        else:

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
        "stderr": all_trial_X_stderr,
        "tstat": all_trial_X_tstat,
        "mse_btw": all_trial_X_mse_between,
        "mean": all_trial_X_hrf_ts_weighted,
        "mse_within": all_trial_X_mse_within,
    }

    return results

# %%
