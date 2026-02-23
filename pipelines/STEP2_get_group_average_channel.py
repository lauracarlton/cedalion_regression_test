"""
Usage
-----
Run after all per-subject image reconstructions have completed::

        python FIG8_STEP3_get_group_average.py

Configurables (near top of file)
--------------------------------
- ROOT_DIR: dataset root containing BIDS-like subject folders.
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

from cedalion import units, xrutils, nirs, io
from cedalion.io.forward_model import load_Adot

def run_pipeline_channel_group_avg(
                                    ROOT_DIR, 
                                    TASK="BS", 
                                    NOISE_MODEL='ar_irls', 
                                    REC_STR='conc', 
                                    TRIAL_TYPES=['right', 'left'],
                                    GOLDEN_CHANNEL='S53D28'
                            ):

    PROBE_DIR = os.path.join(ROOT_DIR, 'probe')
    dirs = os.listdir(ROOT_DIR)
    subject_list = [d for d in dirs if "sub" in d]
    # define cfg_mse 
    cfg_mse = {"mse_val_for_bad_data": 1e7*units.micromolar**2, "mse_amp_thresh": 1e-3 * units.V, "blockaverage_val": 0, "mse_min_thresh": 1e0 * units.micromolar**2}

    # load in a reference snirf file
    with open(ROOT_DIR + '/probe/geo3d.pkl', 'rb') as f:
        geo3d = pickle.load(f)

    E = nirs.get_extinction_coefficients("prahl", [760, 850])
    Einv = xrutils.pinv(E)
    dpf = xr.DataArray([1, 1], dims="wavelength", coords={"wavelength": [760, 850]})

    all_trial_X_hrf_ts_weighted = None
    for trial_type in TRIAL_TYPES:
        all_subj_X_hrf_ts = []
        all_subj_X_mse = []
        print(f"\ttrial_type - {trial_type}")

        for subj in subject_list:

            with gzip.open(os.path.join(
                        ROOT_DIR,
                        'tmp',
                        subj,
                        "processed_data",
                        f"{subj}_task-{TASK}_{REC_STR}_hrf_estimates_{NOISE_MODEL}.pkl.gz"
                ),"rb") as f:
                    all_results = pickle.load(f)


            subj_hrf = all_results["hrf_per_subj"]
            subj_mse = all_results["hrf_mse_per_subj"]
            bad_channels = all_results["bad_indices"]

            conc_hrf = subj_hrf.squeeze().sel(trial_type=trial_type)
            conc_hrf = conc_hrf.pint.to('micromolar')

            mse = subj_mse.squeeze().sel(trial_type=trial_type)
            mse = mse.pint.to('micromolar**2')
            channels = conc_hrf.channel
            mse.loc[channels.isel(channel=bad_channels),:, :] = cfg_mse["mse_val_for_bad_data"]
            mse = xr.where(
                mse < cfg_mse["mse_min_thresh"], cfg_mse["mse_min_thresh"], mse
            )  # !!! maybe can be removed when we have the between subject mse
            
            conc_hrf.loc[channels.isel(channel=bad_channels),:, :] = cfg_mse["blockaverage_val"]*units.micromolar

            all_subj_X_hrf_ts.append(conc_hrf)
            all_subj_X_mse.append(mse)

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

        
        if all_trial_X_hrf_ts_weighted is None:

            all_trial_X_hrf_ts_weighted = X_hrf_ts_mean_weighted
            all_trial_X_stderr = X_stderr_weighted
            all_trial_X_tstat = X_tstat
            all_trial_X_mse_between = X_mse_weighted_between_subjects
            all_trial_X_mse_within = X_mse_mean_within_subject
        else:

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
        "stderr": all_trial_X_stderr.sel(channel=GOLDEN_CHANNEL).pint.to('molar'),
        "tstat": all_trial_X_tstat.sel(channel=GOLDEN_CHANNEL),
        "mse_btw": all_trial_X_mse_between.sel(channel=GOLDEN_CHANNEL).pint.to('molar**2'),
        "mean": all_trial_X_hrf_ts_weighted.sel(channel=GOLDEN_CHANNEL).pint.to('molar'),
        "mse_within": all_trial_X_mse_within.sel(channel=GOLDEN_CHANNEL).pint.to('molar**2'),
    }

    return results

    # %%
