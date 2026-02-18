#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIG7_STEP2_image_recon_on_HRF.py

Image reconstruction applied to per-subject HRF estimates produced by
FIG8_STEP1_hrf_estimation.py.

This script performs per-subject image reconstruction across several
regularization configurations (direct/indirect and spatial basis (SB) options).
For each subject it:

- Loads precomputed HRF estimates and MSE (gzipped pickle produced by STEP1).
- Converts concentration HRFs to optical density (OD) using subject geometry.
- Computes measurement-space noise (C_meas) from channel MSE and uses it to
    weight image reconstruction.
- Runs image reconstruction (via project helper `image_recon_func`) to produce
    voxel/time images and per-image noise estimates.
- Saves G-matrix basis files for later reuse if they don't already exist.

Usage
-----
- Configure the runtime variables in the CONFIG section below (ROOT_DIR,
    PROBE_DIR, NOISE_MODEL, flags, regularization configs).
- Run from the command line:

        python FIG8_STEP2_image_recon_on_HRF.py

Inputs
------
- Root dataset folder containing per-subject `nirs` subfolders and STEP1
    outputs located under `<ROOT_DIR>/derivatives/cedalion/processed_data/<subject>/`.
- Forward model `Adot.nc` in the PROBE_DIR used for sensitivity and masking.

Configurables (defaults shown)
-----------------------------
- ROOT_DIR (str): '.../BS_bids'
        - Path to BIDS-like dataset used by STEP1 outputs and subject SNIRF files.
- NOISE_MODEL (str): 'ar_irls'
        - Noise-model label used for solving the GLM when reading per-subject STEP1 outputs.
- REC_STR (str): 'conc_o'
        - Record string pointing to concentration data used by STEP1.
- TASK (str): 'BS'
    - Task identifier used to build file IDs.
- CMEAS_FLAG (bool): True
    - Whether to use measured C_meas when performing image reconstruction.
- MAG_TS_FLAG (str): 'MAG'  # expected values: 'MAG' or 'TS'
        - Controls whether to reduce HRF to a single magnitude value (MAG) or
            keep time series (TS) when reconstructing.
- T_WIN (list[int]): [5, 8]
        - Time window (seconds) used for computing magnitude when MAG flag is set.
- HEAD_MODEL (str): 'ICBM152'
    - head model used for generating sensitivity profile 
- lambda_R (float): 1e-6
    - regularization parameter used to scale spatial prior in reconstructions.
- optional_flag (str): ''
    - Additional string appended to output filenames (e.g. for noting special cases).
- EXCLUDED (list): list of subjects to be exluded from analysis
- cfg_mse (dict): keys for MSE masking and thresholds (see script for defaults).
- cfg_list (list[dict]): Regularization configurations evaluated (alpha_meas,
    alpha_spatial, DIRECT, SB, sigma_brain, sigma_scalp).

Outputs
-------
- For each subject and configuration, a gzipped pickle saved to:
    `<ROOT_DIR>/derivatives/processed_data/image_space/<subject>/` containing:
    - 'X_hrf' : reconstructed image(s) (xarray)
    - 'X_mse' : per-image MSE estimates (xarray)

Assumptions and dependencies
----------------------------
- Requires the `cedalion` package and the project's helper modules:
    `image_recon_func` and `spatial_basis_funs` (sys.path is extended in the
    script to load these).
- Uses xarray and pint-aware arrays returned by `cedalion` operations.

Author: Laura Carlton
"""
# %%
import os
import sys
import gzip
import pickle
import warnings

import xarray as xr
from cedalion import nirs, units, io
from cedalion.io.forward_model import load_Adot

# import my own functions from a different directorys
sys.path.append("/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/")
import image_recon_func as irf  
import spatial_basis_func as sbf

# Turn off all warnings
warnings.filterwarnings("ignore")

# %% set up config parameters
ROOT_DIR = os.path.join("/projectnb", "nphfnirs", "s", "datasets", "BSMW_Laura_Miray_2025", "BS_bids_v2")
NOISE_MODEL = "ar_irls"
TASK = "BS"
REC_STR = "conc_o"
CMEAS_FLAG = True
MAG_TS_FLAG = "MAG"  # expected values: 'MAG' or 'TS' (case-sensitive in downstream checks)
T_WIN = [5, 8]
HEAD_MODEL = 'ICBM152'
EXCLUDED = []
lambda_R = 1e-6
optional_flag = ''

cfg_list = [
    {"alpha_meas": 1e1, "alpha_spatial": 1e-3, "lambda_R": lambda_R, "DIRECT": False, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e1, "alpha_spatial": 1e-3, "lambda_R": lambda_R, "DIRECT": True, "SB": False, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e1, "alpha_spatial": 1e-2, "lambda_R": lambda_R, "DIRECT": False, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
    {"alpha_meas": 1e1, "alpha_spatial": 1e-2, "lambda_R": lambda_R, "DIRECT": True, "SB": True, "sigma_brain": 1, "sigma_scalp": 5},
]

cfg_mse = {"mse_val_for_bad_data": 1e1, "mse_amp_thresh": 1e-3 * units.V, "blockaverage_val": 0, "mse_min_thresh": 1e-6}

dirs = os.listdir(ROOT_DIR)
subject_list = [d for d in dirs if "sub" in d and d not in EXCLUDED]

PROBE_DIR = os.path.join(ROOT_DIR, "derivatives", "cedalion", "fw", 'probe')

# %% load head model
head, PARCEL_DIR = irf.load_head_model(HEAD_MODEL, with_parcels=True)
Adot = load_Adot(os.path.join(PROBE_DIR, "Adot.nc"))

# %% run image recon
"""
do the image reconstruction of each subject independently
- this is the unweighted subject block average magnitude
- then reconstruct their individual MSE
- then get the weighted average in image space
- get the total standard error using between + within subject MSE
"""

for cfg in cfg_list:
    F = None
    D = None
    max_eig = None

    DIRECT = cfg["DIRECT"]
    SB = cfg["SB"]

    sigma_brain = cfg["sigma_brain"]
    sigma_scalp = cfg["sigma_scalp"]
    alpha_meas = cfg["alpha_meas"]
    alpha_spatial = cfg["alpha_spatial"]
    lambda_R = cfg["lambda_R"]

    if os.path.exists(os.path.join(PROBE_DIR, f"G_matrix_sigmabrain-{float(sigma_brain)}.pkl")) and os.path.exists(
        os.path.join(PROBE_DIR, f"G_matrix_sigmascalp-{float(sigma_scalp)}.pkl")):
        G_EXISTS = True
        with open(os.path.join(PROBE_DIR, f"G_matrix_sigmabrain-{float(sigma_brain)}.pkl"), "rb") as f:
            G_brain = pickle.load(f)

        with open(os.path.join(PROBE_DIR, f"G_matrix_sigmascalp-{float(sigma_scalp)}.pkl"), "rb") as f:
            G_scalp = pickle.load(f)

        G = {"G_brain": G_brain, "G_scalp": G_scalp}

    else:
        G_EXISTS = False
        G = None

    if DIRECT:
        direct_name = "direct"
    else:
        direct_name = "indirect"

    if CMEAS_FLAG:
        Cmeas_name = "Cmeas"
    else:
        Cmeas_name = "noCmeas"

    cfg_sbf = {
        "mask_threshold": -2,
        "threshold_brain": sigma_brain * units.mm,
        "threshold_scalp": sigma_scalp * units.mm,
        "sigma_brain": sigma_brain * units.mm,
        "sigma_scalp": sigma_scalp * units.mm,
    }

    print(f"alpha_meas = {alpha_meas}, alpha_spatial = {alpha_spatial}, SB = {SB}, {direct_name}")

    for subject in subject_list:
        all_trial_X_hrf = None
        all_trial_X_mse = None
        SAVE_DIR = os.path.join(ROOT_DIR, "derivatives", "cedalion", "processed_data", "image_space", subject)
        os.makedirs(SAVE_DIR, exist_ok=True)

        recordings = io.read_snirf(os.path.join(ROOT_DIR, subject, "nirs", f"{subject}_task-{TASK}_run-01_nirs.snirf"))
        rec = recordings[0]
        geo3d = rec.geo3d
        amp = rec["amp"]
        # create wavelength-dependent helpers now that `amp` is available
        dpf = xr.DataArray([1, 1], dims="wavelength", coords={"wavelength": amp.wavelength})
        E = nirs.get_extinction_coefficients("prahl", amp.wavelength)
        meas_list = rec._measurement_lists["amp"]

        print("Loading saved data")
        with gzip.open(
            os.path.join(
                ROOT_DIR,
                "derivatives",
                "cedalion",
                "processed_data",
                subject,
                f"{subject}_task-{TASK}_{REC_STR}_hrf_estimates_{NOISE_MODEL}.pkl.gz"
            ),
            "rb",
        ) as f:
            all_results = pickle.load(f)

        subj_hrf = all_results["hrf_per_subj"]
        subj_mse = all_results["hrf_mse_per_subj"]
        bad_channels = all_results["bad_indices"]

        print(f"\tCalculating subject = {subject}")

        for trial_type in subj_hrf.trial_type:

            print(f"\t\tGetting images for trial type = {trial_type.values}")

            hrf = subj_hrf.squeeze().sel(trial_type=trial_type)

            od_hrf = nirs.conc2od(hrf, geo3d, dpf)

            mse = subj_mse.squeeze().sel(trial_type=trial_type)
            od_mse = xr.dot(E**2, mse, dim=["chromo"]) * 1 * units.mm**2

            channels = od_hrf.channel
            od_mse.loc[:, channels.isel(channel=bad_channels), :] = cfg_mse["mse_val_for_bad_data"]
            od_mse = xr.where(
                od_mse < cfg_mse["mse_min_thresh"], cfg_mse["mse_min_thresh"], od_mse
            )  # !!! maybe can be removed when we have the between subject mse
            od_hrf.loc[:, channels.isel(channel=bad_channels), :] = cfg_mse["blockaverage_val"]

            od_mse_mag = od_mse.mean("time")

            if MAG_TS_FLAG == "MAG":
                od_hrf = od_hrf.sel(time=slice(T_WIN[0], T_WIN[1])).mean("time")
                fname_flag = "mag"
            else:
                fname_flag = "ts"

            C_meas = od_mse_mag.pint.dequantify()
            C_meas = C_meas.stack(measurement=("channel", "wavelength")).sortby("wavelength")

            X_hrf, W, D, F, G, max_eig = irf.do_image_recon(
                                        od_hrf,
                                        head=head,
                                        Adot=Adot,
                                        C_meas_flag=CMEAS_FLAG,
                                        C_meas=C_meas,
                                        wavelength=Adot.wavelength,
                                        DIRECT=DIRECT,
                                        SB=SB,
                                        cfg_sbf=cfg_sbf,
                                        alpha_spatial=alpha_spatial,
                                        alpha_meas=alpha_meas,
                                        lambda_R=lambda_R,
                                        F=F,
                                        D=D,
                                        G=G,
                                        max_eig=max_eig
                                    )

            if SB and not G_EXISTS:
                with open(os.path.join(PROBE_DIR, f"G_matrix_sigmabrain-{float(sigma_brain)}.pkl"), "wb") as f:
                    pickle.dump(G["G_brain"], f)

                with open(os.path.join(PROBE_DIR, f"G_matrix_sigmascalp-{float(sigma_scalp)}.pkl"), "wb") as f:
                    pickle.dump(G["G_scalp"], f)

                G_EXISTS = True

            if DIRECT:
                Adot_tmp = irf.get_Adot_scaled(Adot, Adot.wavelength)
                if SB:
                    Adot_tmp = sbf.get_H_stacked(G, Adot_tmp)

            else: 
                Adot_tmp = Adot
                if SB:
                    Adot_tmp = sbf.get_H(G, Adot_tmp)
                    
            X_mse = irf.get_image_noise_posterior(Adot_tmp, 
                                                W, 
                                                alpha_spatial=alpha_spatial, 
                                                lambda_R=lambda_R,
                                                DIRECT=DIRECT, 
                                                SB=SB, 
                                                G=G)
            if all_trial_X_hrf is None:

                all_trial_X_hrf = X_hrf
                all_trial_X_hrf = all_trial_X_hrf.assign_coords(trial_type=trial_type)

                all_trial_X_mse = X_mse
                all_trial_X_mse = all_trial_X_mse.assign_coords(trial_type=trial_type)

            else:

                X_hrf = X_hrf.assign_coords(trial_type=trial_type)
                X_mse = X_mse.assign_coords(trial_type=trial_type)

                all_trial_X_hrf = xr.concat([all_trial_X_hrf, X_hrf], dim="trial_type")
                all_trial_X_mse = xr.concat([all_trial_X_mse, X_mse], dim="trial_type")

        results = {"X_hrf": all_trial_X_hrf, "X_mse": all_trial_X_mse}

        print(f"\t\tSaving to {SAVE_DIR}")

        if SB:
            filepath = os.path.join(
                SAVE_DIR,
                f"{subject}_task-{TASK}_image_hrf_{fname_flag}_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{Cmeas_name}_{NOISE_MODEL}{optional_flag}.pkl.gz",
            )
        else:
            filepath = os.path.join(
                SAVE_DIR,
                f"{subject}_task-{TASK}_image_hrf_{fname_flag}_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_{direct_name}_{Cmeas_name}_{NOISE_MODEL}{optional_flag}.pkl.gz",
            )

        file = gzip.GzipFile(filepath, "wb")
        file.write(pickle.dumps(results))
        file.close()

    print("Job Complete")
    # %%
