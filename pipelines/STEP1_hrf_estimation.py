#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
Author: Laura Carlton
"""
# %% Imports
##############################################################################
import os
import gzip
import pickle
import sys
import warnings

import pandas as pd
import numpy as np
import xarray as xr

from cedalion import units, nirs, io, xrutils
from cedalion.io.forward_model import load_Adot
from cedalion.sigproc import motion, quality
from cedalion.sigproc.frequency import freq_filter

# sys.path.append("/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/cedalion_regression_test/modules")
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PIPELINES_DIR = os.path.join(PROJECT_ROOT, "modules")
sys.path.insert(0, PIPELINES_DIR)

import processing_func as pf

# Turn off all warnings
warnings.filterwarnings("ignore")

# %% Initial root directory and analysis parameters
def run_pipeline_estimate_hrf(
                                ROOT_DIR,
                                TASK="BS",
                                N_RUNS=3,
                                NOISE_MODEL="ar_irls",
                                SAVE_DIR=None,
                                GOLDEN_CHANNEL='S53D28'
                            ):
    """
    Run per-subject HRF estimation pipeline from raw fNIRS data.
    
    Performs complete preprocessing and GLM-based hemodynamic response function
    estimation for all subjects in a BIDS-like dataset. Processes multiple runs
    per subject, applies quality control, motion correction (OLS only), filtering,
    and GLM fitting with configurable noise models.
    
    Parameters
    ----------
    ROOT_DIR : str
        Root directory containing subject folders in BIDS-like structure.
        Expected structure: ROOT_DIR/sub-XXX/nirs/sub-XXX_task-*_run-*_nirs.snirf
    TASK : str, optional
        Task identifier used in filename construction (default: "BS").
    N_RUNS : int, optional
        Number of runs to process per subject (default: 3).
    NOISE_MODEL : str, optional
        GLM noise model: 'ols' or 'ar_irls' (default: "ar_irls").
        - 'ols': Uses TDDR, bandpass filtering, polynomial drift
        - 'ar_irls': No TDDR/filtering, Legendre drift, robust to autocorrelation
    SAVE_DIR : str, optional
        Directory for saving intermediate results. If None, uses ROOT_DIR/tmp.
    GOLDEN_CHANNEL : str, optional
        Reference channel for regression testing (default: 'S53D28').
    
    Returns
    -------
    hrf : xr.DataArray
        HRF estimate for golden channel with dimensions (trial_type, chromo, time).
        Units: molar (M).
    mse : xr.DataArray
        Mean squared error for golden channel HRF with dimensions
        (trial_type, chromo, time). Units: molar^2 (M^2).
    
    Notes
    -----
    Pipeline steps:
    1. Load amplitude data and events
    2. Apply median filtering (window=3)
    3. Prune channels based on SNR, amplitude, SD distance
    4. Convert to optical density
    5. Apply TDDR motion correction (OLS only)
    6. Bandpass filter 0-0.5 Hz (OLS only)
    7. Convert to concentration (HbO, HbR)
    8. Fit GLM with HRF, drift, and short-separation regressors
    9. Save per-subject results to SAVE_DIR/subject/processed_data/
    
    Configuration is constructed internally based on NOISE_MODEL parameter.
    
    Outputs saved per subject:
    - {subject}_task-{TASK}_conc_hrf_estimates_{NOISE_MODEL}.pkl.gz
    
    Example
    -------
    >>> hrf, mse = run_pipeline_estimate_hrf(
    ...     ROOT_DIR="/data/BS_bids",
    ...     TASK="BS",
    ...     N_RUNS=3,
    ...     NOISE_MODEL="ar_irls",
    ...     SAVE_DIR="/data/output"
    ... )
    """

    dirs = os.listdir(ROOT_DIR)
    subject_list = [d for d in dirs if "sub" in d]

    PROBE_DIR = os.path.join(ROOT_DIR, "probe")

    if NOISE_MODEL == "ols":
        DO_TDDR = True
        DO_DRIFT = True
        DO_DRIFT_LEGENDRE = False
        DRIFT_ORDER = 3
        F_MIN = 0 * units.Hz
        F_MAX = 0.5 * units.Hz
    elif NOISE_MODEL == "ar_irls":
        DO_TDDR = False
        DO_DRIFT = False
        DO_DRIFT_LEGENDRE = True
        DRIFT_ORDER = 3
        F_MAX = 0
        F_MIN = 0
    else:
        print("Not a valid noise model - please select ols or ar_irls")

    cfg_GLM = {
        "do_drift": DO_DRIFT,
        "do_drift_legendre": DO_DRIFT_LEGENDRE,
        "do_short_sep": True,
        "drift_order": DRIFT_ORDER,
        "distance_threshold": 20 * units.mm,  # for ssr
        "short_channel_method": "mean",
        "noise_model": NOISE_MODEL,
        "t_delta": 1 * units.s,  # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
        "t_std": 1 * units.s,
        "t_pre": 2 * units.s,
        "t_post": 10 * units.s,
    }

    cfg_dataset = {
        "root_dir": ROOT_DIR,
        "subj_ids": subject_list,
        "file_ids": [f"{TASK}_run-0{i}" for i in range(1, N_RUNS + 1)],
    }

    cfg_prune = {
        "snr_thresh": 5,  # the SNR (std/mean) of a channel.
        "sd_thresh": [1, 40] * units.mm,  # defines the lower and upper bounds for the source-detector separation that we would like to keep
        "amp_thresh": [1e-5, 0.84] * units.V,  # define whether a channel's amplitude is within a certain range
        "perc_time_clean_thresh": 0.6,
        "sci_threshold": 0.6,
        "psp_threshold": 0.1,
        "window_length": 5 * units.s,
        "flag_use_sci": False,
        "flag_use_psp": False,
    }

    cfg_bandpass = {"fmin": F_MIN, "fmax": F_MAX}

    # values for manual adjustment of channel space MSE in OD
    cfg_mse = {"mse_val_for_bad_data": 1e1, "mse_amp_thresh": 1e-3 * units.V, "blockaverage_val": 0, "mse_min_thresh": 1e-6}

    # geo3d = xr.load_dataarray(ROOT_DIR + '/probe/geo3d.nc')
    with open(os.path.join(PROBE_DIR, "geo3d.pkl"), 'rb') as f:
        geo3d = pickle.load(f)
    
    #% RUN PREPROCESSING
    # loop over subjects and files
    for ss, subject in enumerate(subject_list):

        print(f"Processing subject {ss+1} of {len(subject_list)}")
        SAVE_DIR_tmp = os.path.join(SAVE_DIR, subject, 'processed_data')
        os.makedirs(SAVE_DIR_tmp, exist_ok=True)

        print("\tRUNNING PREPROCESSING")
        for file_idx in range(N_RUNS):

            filenm = f"{subject}_task-{cfg_dataset['file_ids'][file_idx]}"
            print(f"\t\tProcessing  {file_idx+1} of {N_RUNS} files : {filenm}")

            subDir = os.path.join(cfg_dataset["root_dir"], subject, "nirs")

            file_path = os.path.join(subDir, filenm)
            amp = xr.load_dataarray(file_path + '_sample.nc')
            wavelengths = amp.wavelength

            stim_df = pd.read_csv(file_path + "_events.tsv", sep="\t")

            amp = amp.where(~amp.isnull(), 1e-18)
            amp = amp.where(amp > 0, 1e-18)

            # if first value is 1e-18 then replace with second value
            indices = np.where(amp[:, 0, 0] == 1e-18)
            amp[indices[0], 0, 0] = amp[indices[0], 0, 1]
            indices = np.where(amp[:, 1, 0] == 1e-18)
            amp[indices[0], 1, 0] = amp[indices[0], 1, 1]

            amp = amp.pint.dequantify().pint.quantify("V")

            # prune channels 
            amp_thresh_sat = [0.*units.V, cfg_prune['amp_thresh'][1]]
            amp_thresh_low = [cfg_prune['amp_thresh'][0], 1*units.V]
            _, amp_mask_sat = quality.mean_amp( amp,  amp_thresh_sat)
            _, amp_mask_low = quality.mean_amp( amp,  amp_thresh_low)
            _, snr_mask = quality.snr( amp,  cfg_prune['snr_thresh'])            
            chs_pruned = xr.DataArray(np.zeros(amp.shape[0]), dims=["channel"], coords={"channel": amp.channel})
            #initialize chs_pruned to 0.4
            chs_pruned[:] = 0.4
            chs_pruned[~snr_mask[:,0]] = 0.19
            chs_pruned[~amp_mask_sat[:,0]] = 0
            chs_pruned[~amp_mask_low[:,0]] = 0.8

            dpf = xr.DataArray(
                [1, 1],
                dims="wavelength",
                coords={"wavelength": amp.wavelength},
            )

            E = nirs.get_extinction_coefficients('prahl', wavelengths)
            Einv = xrutils.pinv(E)

            od = nirs.cw.int2od(amp)
            od.time.attrs["units"] = units.s

            if DO_TDDR:
                od = motion.tddr(od)

            od = od.where(~od.isnull(), 1e-18)

            if cfg_bandpass["fmin"] > 0 or cfg_bandpass["fmax"] > 0:
                od = freq_filter(od, cfg_bandpass["fmin"], cfg_bandpass["fmax"])

            conc = xr.dot(Einv, od / (dpf * 1*units.mm), dim=["wavelength"])
            conc.time.attrs["units"] = units.s

            if file_idx == 0:
                all_runs = []
                all_chs_pruned = []
                all_stims = []

                all_runs.append(conc)
                all_chs_pruned.append(chs_pruned)
                all_stims.append(stim_df)

            else:
                all_runs.append(conc)
                all_chs_pruned.append(chs_pruned)
                all_stims.append(stim_df)

        print("\tESTIMATE THE HRF")
        results, hrf_estimate, hrf_mse = pf.GLM(all_runs, cfg_GLM, geo3d, all_chs_pruned, all_stims)

        # get the indices for bad channels
        n_chs = len(amp.channel)

        hrf_estimate = hrf_estimate.transpose("channel", "time", "chromo", "trial_type")
        hrf_estimate = hrf_estimate - hrf_estimate.sel(time=(hrf_estimate.time < 0)).mean("time")

        hrf_mse = hrf_mse.transpose("channel", "time", "chromo", "trial_type")

        hrf_per_subj = hrf_estimate.expand_dims("subj")
        hrf_per_subj = hrf_per_subj.assign_coords(subj=[subject])

        hrf_mse_per_subj = hrf_mse.expand_dims("subj")
        hrf_mse_per_subj = hrf_mse_per_subj.assign_coords(subj=[subject])

        n_chs = len(amp.channel)
        idx_amp = np.where(amp.mean('time') < cfg_mse["mse_amp_thresh"])[0]
        idx_sat = np.where(all_chs_pruned[0] == 0.0)[0]
        bad_indices = np.unique(np.concat([idx_amp, idx_sat]))
        
        print("\tHRF estimation complete")

        # save per subject results concentration and then image recon will take and convert to OD
        file_path_pkl = os.path.join(SAVE_DIR_tmp, f"{subject}_task-{TASK}_conc_hrf_estimates_{NOISE_MODEL}.pkl.gz")

        # save the individual results to a pickle file for image recon
        file = gzip.GzipFile(file_path_pkl, "wb")

        all_results = {
            "hrf_per_subj": hrf_per_subj,  # always unweighted   - load into img recon
            "hrf_mse_per_subj": hrf_mse_per_subj,  # - load into img recon
            "bad_indices": bad_indices,
        }

        file.write(pickle.dumps(all_results))
        file.close()

        # if SAVE_RESIDUAL:
        #     file_path_pkl = os.path.join(SAVE_DIR, f"{subject}_task-{TASK}_{REC_STR}_glm_residual_{NOISE_MODEL}.pkl")

        #     residual = results.sm.resid
        #     with open(file_path_pkl, "wb") as f:
        #         pickle.dump(residual, f)
        if subject == 'sub-618':
            single_sub_hrf = hrf_per_subj.sel(channel=GOLDEN_CHANNEL)
            single_sub_mse = hrf_mse_per_subj.sel(channel=GOLDEN_CHANNEL)

        print("Saved individual HRF to " + file_path_pkl)

    return single_sub_hrf, single_sub_mse

    # %%
