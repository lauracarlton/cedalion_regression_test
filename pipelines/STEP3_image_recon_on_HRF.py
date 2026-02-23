#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Laura Carlton
"""
# %%
import os
import sys
import gzip
import pickle
import warnings

import xarray as xr
from cedalion import units, data, io, nirs
from cedalion.nirs import cw
from cedalion import dataclasses as cdc
from cedalion.dot import head_model as hm
from cedalion.dot import forward_model as fm
from cedalion.dot import image_recon as ir
from cedalion.io.forward_model import load_Adot

# Turn off all warnings
warnings.filterwarnings("ignore")

# %% set up config parameters

def run_pipeline_image_recon(
                            ROOT_DIR, 
                            cfg,
                            TASK='BS', 
                            NOISE_MODEL='ar_irls', 
                            REC_STR='conc', 
                            T_WIN=[5,8],
                        ):
    """
    Reconstruct brain images from per-subject channel-space HRF estimates.
    
    Performs DOT (Diffuse Optical Tomography) image reconstruction using a forward
    model and configurable regularization. Converts concentration/optical density
    channel measurements to spatially resolved cortical activation images.
    
    Parameters
    ----------
    ROOT_DIR : str
        Root directory containing STEP1 outputs and probe information.
        Expected: ROOT_DIR/tmp/sub-XXX/processed_data/*.pkl.gz
    cfg : dict
        Reconstruction configuration with keys:
        - 'method' : str, 'conc' (direct) or 'mua2conc' (indirect)
        - 'alpha_meas' : float, measurement regularization (e.g., 1e4)
        - 'alpha_spatial' : float, spatial prior strength (e.g., 1e-3)
        - 'lambda_R' : float, depth regularization (e.g., 1e-6)
        - 'SB' : bool, use spatial basis functions (Gaussian kernels)
        - 'sigma_brain' : float, brain basis width in mm (if SB=True)
        - 'sigma_scalp' : float, scalp basis width in mm (if SB=True)
    TASK : str, optional
        Task identifier (default: 'BS').
    NOISE_MODEL : str, optional
        Noise model label: 'ols' or 'ar_irls' (default: 'ar_irls').
    REC_STR : str, optional
        Recording type: 'conc' for concentration (default: 'conc').
    T_WIN : list of float, optional
        Time window [start, end] in seconds for HRF magnitude computation
        (default: [5, 8]).
    
    Returns
    -------
    image : xr.DataArray
        Reconstructed image for first subject with dimensions
        (trial_type, chromo, flatmap or vertex).
        Units: molar (M).
    post_mse : xr.DataArray
        Posterior MSE (uncertainty) for first subject with same dimensions.
        Units: molar^2 (M^2).
    
    Notes
    -----
    Reconstruction methods:
    - 'conc' (direct): Minimizes ||AΔc - ΔOD||² for concentration changes
    - 'mua2conc' (indirect): Reconstructs absorption coefficient, then converts
    
    Regularization:
    - Measurement term: α_meas × C_meas⁻¹
    - Spatial prior: α_spatial × C_spatial⁻¹ (depth-weighted by λ_R)
    - Spatial basis: Reduces dimensionality via Gaussian kernels (if SB=True)
    
    Time window:
    - HRF is averaged over T_WIN to produce magnitude images
    - Reduces temporal dimension for spatial reconstruction
    
    Head model:
    - Uses ICBM152 brain/scalp surface templates
    - Forward model (Adot) maps cortex → channel sensitivity
    
    Outputs saved per subject:
    - {subject}_task-{TASK}_image_hrf_mag_*.pkl.gz
    
    Example
    -------
    >>> cfg = {
    ...     "method": "conc",
    ...     "alpha_meas": 1e4,
    ...     "alpha_spatial": 1e-3,
    ...     "lambda_R": 1e-6,
    ...     "SB": True,
    ...     "sigma_brain": 1,
    ...     "sigma_scalp": 5
    ... }
    >>> image, mse = run_pipeline_image_recon(
    ...     ROOT_DIR="/data/BS_bids",
    ...     cfg=cfg,
    ...     TASK="BS",
    ...     T_WIN=[5, 8]
    ... )
    """

    cfg_mse = {"mse_val_for_bad_data": 1e1, "mse_amp_thresh": 1e-3 * units.V, "blockaverage_val": 0, "mse_min_thresh": 1e-6}

    dirs = os.listdir(ROOT_DIR)
    subject_list = [d for d in dirs if "sub" in d]

    PROBE_DIR = os.path.join(ROOT_DIR, 'probe')

    # load head model
    SEG_DATADIR, mask_files, landmarks_file = data.get_icbm152_segmentation()
    PARCEL_DIR = data.get_icbm152_parcel_file()
    head = hm.TwoSurfaceHeadModel.from_surfaces(
    segmentation_dir=SEG_DATADIR,
    mask_files = mask_files,
    brain_surface_file= os.path.join(SEG_DATADIR, "mask_brain.obj"),
    scalp_surface_file= os.path.join(SEG_DATADIR, "mask_scalp.obj"),
    landmarks_ras_file=landmarks_file,
    smoothing=0.5,
    fill_holes=True,
    # parcel_file=PARCEL_DIR
    ) 
    head.scalp.units = units.mm
    head.brain.units = units.mm

    Adot = load_Adot(os.path.join(PROBE_DIR, "Adot.nc"))
    with open(ROOT_DIR + '/probe/geo3d.pkl', 'rb') as f:
        geo3d = pickle.load(f)
        
    # run image recon
    """
    do the image reconstruction of each subject independently
    - this is the unweighted subject block average magnitude
    - then reconstruct their individual MSE
    - then get the weighted average in image space
    - get the total standard error using between + within subject MSE
    """

    method = cfg["method"]
    SB = cfg["SB"]

    sigma_brain = cfg["sigma_brain"]
    sigma_scalp = cfg["sigma_scalp"]
    alpha_meas = cfg["alpha_meas"]
    alpha_spatial = cfg["alpha_spatial"]
    lambda_R = cfg["lambda_R"]

    if cfg['SB']:
        sbf = ir.OriginalGaussianSpatialBasisFunctions(head, 
                                Adot,
                                threshold_brain=sigma_brain*units.mm,
                                threshold_scalp=sigma_scalp*units.mm,
                                sigma_brain=sigma_brain*units.mm,
                                sigma_scalp=sigma_scalp*units.mm,
                                mask_threshold=-2
                        )
    else:
        sbf = None

    imagerecon_obj = ir.ImageRecon(
                                Adot,
                                recon_mode=method,
                                alpha_meas=alpha_meas, 
                                alpha_spatial=alpha_spatial, 
                                lambda_R_conc = lambda_R,
                                apply_c_meas=True,
                                spatial_basis_functions=sbf
                                )

    if method == 'conc':
        direct_name = "direct"
    else:
        direct_name = "indirect"

    print(f"alpha_meas = {alpha_meas}, alpha_spatial = {alpha_spatial}, SB = {SB}, {direct_name}")

    for subject in subject_list:
        all_trial_X_hrf = None
        all_trial_X_mse = None
        SAVE_DIR = os.path.join(ROOT_DIR, 'tmp', subject, "processed_data")
        os.makedirs(SAVE_DIR, exist_ok=True)

        # create wavelength-dependent helpers now that `amp` is available
        dpf = xr.DataArray([1, 1], dims="wavelength", coords={"wavelength": Adot.wavelength})
        E = nirs.get_extinction_coefficients("prahl", Adot.wavelength)

        print("Loading saved data")
        with gzip.open(
            os.path.join(
                ROOT_DIR,
                'tmp',
                subject,
                "processed_data",
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

            od_hrf = nirs.cw.conc2od(hrf, geo3d, dpf)

            mse = subj_mse.squeeze().sel(trial_type=trial_type)
            od_mse = xr.dot(E**2, mse, dim=["chromo"]) * 1 * units.mm**2

            channels = od_hrf.channel
            od_mse.loc[:, channels.isel(channel=bad_channels), :] = cfg_mse["mse_val_for_bad_data"]
            od_mse = xr.where(
                od_mse < cfg_mse["mse_min_thresh"], cfg_mse["mse_min_thresh"], od_mse
            )  # !!! maybe can be removed when we have the between subject mse
            od_hrf.loc[:, channels.isel(channel=bad_channels), :] = cfg_mse["blockaverage_val"]

            od_mse_mag = od_mse.mean("time")

            od_hrf = od_hrf.sel(time=slice(T_WIN[0], T_WIN[1])).mean("time")

            C_meas = od_mse_mag.pint.dequantify()

            X_hrf = imagerecon_obj.reconstruct(od_hrf, c_meas=C_meas)
            X_hrf = X_hrf.pint.to('molar')
            X_mse = imagerecon_obj.get_image_noise_posterior(C_meas)
            X_mse = X_mse.pint.to('molar**2')

            if 'parcel' in Adot.coords:
                X_mse = X_mse.assign_coords({"parcel" : ("vertex", Adot.coords['parcel'].values)})
                                
            if 'is_brain' in Adot.coords:
                X_mse = X_mse.assign_coords({"is_brain": ("vertex", Adot.coords['is_brain'].values)}) 

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
                f"{subject}_task-{TASK}_image_hrf_mag_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_sb-{sigma_brain}_ss-{sigma_scalp}_{direct_name}_{NOISE_MODEL}.pkl.gz",
            )
        else:
            filepath = os.path.join(
                SAVE_DIR,
                f"{subject}_task-{TASK}_image_hrf_mag_as-{alpha_spatial:.0e}_ls-{lambda_R:.0e}_am-{alpha_meas:.0e}_{direct_name}_{NOISE_MODEL}.pkl.gz",
            )

        file = gzip.GzipFile(filepath, "wb")
        file.write(pickle.dumps(results))
        file.close()

        if subject == 'sub-618':
            single_sub_image = all_trial_X_hrf
            single_sub_post = all_trial_X_mse


    return single_sub_image, single_sub_post


        # %%
