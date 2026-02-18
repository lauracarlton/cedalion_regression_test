#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
processing_func.py

fNIRS data preprocessing and GLM analysis utilities. This module provides
functions for quality control, channel pruning, temporal filtering, General
Linear Model (GLM) analysis with various noise models, and spatial smoothing
of reconstructed images.

Key Functionality:
- Channel quality control: Prune channels based on amplitude, SNR, and SD distance
- GLM analysis: Fit hemodynamic response functions with drift and short-separation regression
- Run concatenation: Combine multiple runs for group-level analysis
- Spatial smoothing: Apply Gaussian kernels and PCA-based smoothing to images
- Design matrix construction: Build regressors for drift, short channels, and HRF

Supported GLM noise models:
- OLS (Ordinary Least Squares): Requires TDDR motion correction and drift regressors
- AR-IRLS (Autoregressive Iteratively Reweighted Least Squares): Uses Legendre drift

Author: Laura Carlton | lcarlton@bu.edu
Created: January 16, 2025
"""

from functools import reduce
import operator

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial.distance import cdist

import cedalion.models.glm as glm
import cedalion.nirs as nirs
import cedalion.sigproc.frequency as frequency
from cedalion import units
from cedalion.sigproc import quality


def prune_channels(rec, amp_thresh=[1e-3, 0.84]*units.V, sd_thresh=[0, 45]*units.mm, snr_thresh=5):
    """
    Identify and flag poor quality channels based on multiple criteria.
    
    Applies quality control metrics including source-detector distance, mean amplitude,
    and signal-to-noise ratio. Returns a quality score for each channel ranging from
    0.0 (worst) to 1.0 (best) with intermediate values for specific failure modes.
    
    Parameters
    ----------
    rec : dict
        Recording containing 'amp' (amplitude data) and geo3d (geometry).
    amp_thresh : cedalion.Quantity, optional
        [min, max] amplitude thresholds in Volts (default: [1e-3, 0.84] V).
        Channels outside this range are flagged.
    sd_thresh : cedalion.Quantity, optional
        [min, max] source-detector distance thresholds in mm (default: [0, 45] mm).
    snr_thresh : float, optional
        Minimum signal-to-noise ratio threshold (default: 5).
    
    Returns
    -------
    rec : dict
        Updated recording with 'amp_pruned' field containing data with flagged channels.
    chs_pruned : xr.DataArray
        Quality scores for each channel with dimensions (channel,):
        - 0.0: Saturated (amplitude too high) or too low
        - 0.19: Failed SNR threshold
        - 0.4: Passed all tests (default/good)
        - 0.8: Amplitude below minimum but above zero
        
    Notes
    -----
    Uses cedalion.sigproc.quality module for individual metric calculations.
    The quality scores allow downstream functions to weight or exclude channels.
    """
    
    amp_thresh_sat = [0.*units.V, amp_thresh[1]]
    amp_thresh_low = [amp_thresh[0], 1*units.V]

    _, sd_mask = quality.sd_dist( rec['amp'], rec.geo3d,  sd_thresh)
    _, amp_mask = quality.mean_amp( rec['amp'],  amp_thresh)
    _, amp_mask_sat = quality.mean_amp( rec['amp'],  amp_thresh_sat)
    _, amp_mask_low = quality.mean_amp( rec['amp'],  amp_thresh_low)
    _, snr_mask = quality.snr( rec['amp'],  snr_thresh)

    masks = [sd_mask, amp_mask, snr_mask]

    rec['amp_pruned'], drop_list = quality.prune_ch(rec['amp'], masks, "all", flag_drop=False)
    
    chs_pruned = xr.DataArray(np.zeros(rec['amp'].shape[0]), dims=["channel"], coords={"channel": rec['amp'].channel})

    #i initialize chs_pruned to 0.4
    chs_pruned[:] = 0.4
    chs_pruned[~snr_mask[:,0]] = 0.19
    chs_pruned[~amp_mask_sat[:,0]] = 0
    chs_pruned[~amp_mask_low[:,0]] = 0.8

    return rec, chs_pruned


def prune_mask_ts(ts, pruned_chans):
    """
    Mask pruned channels in time series data with NaN values.
    
    Replaces data for specified channels with NaN to exclude them from analysis
    while preserving array structure and coordinates. Supports both (channel, wavelength, time)
    and (chromo, channel, time) dimension orderings.
    
    Parameters
    ----------
    ts : xr.DataArray
        Time series data with dimensions including 'channel' and 'time'.
        Expected shapes:
        - (channel, wavelength, time) for raw optical density data
        - (chromo, channel, time) for concentration data
    pruned_chans : array-like
        List or array of channel labels to mask with NaN.
    
    Returns
    -------
    ts_masked : xr.DataArray
        Time series with specified channels replaced by NaN.
        
    Raises
    ------
    ValueError
        If input shape doesn't match expected (chan, dim, time) or (dim, chan, time) patterns.
        
    Notes
    -----
    Automatically detects dimension ordering and broadcasts mask appropriately.
    Preserves all coordinates and attributes from input.
    """
    mask = np.isin(ts.channel.values, pruned_chans)
    
    if ts.ndim == 3 and ts.shape[0] == len(ts.channel):
        mask_expanded = mask[:, None, None]  # (chan, wav, time)
    elif ts.ndim == 3 and ts.shape[1] == len(ts.channel):
        mask_expanded = mask[None, :, None]  # (chrom, chan, time)
    else:
        raise ValueError("Expected input shape to be either (chan, dim, time) or (dim, chan, time)")

    ts_masked = ts.where(~mask_expanded, np.nan)
    return ts_masked

def median_filter(rec, median_filt = 3):
    """
    Apply temporal median filter to amplitude data.
    
    Applies a rolling median filter along the time dimension to reduce spike artifacts
    while preserving signal edges better than mean filtering. Uses edge padding to
    avoid boundary artifacts.
    
    Parameters
    ----------
    rec : dict
        Recording dictionary containing 'amp' field with time series data.
    median_filt : int, optional
        Window size for median filter in samples (default: 3).
        Should be odd for symmetric filtering.
    
    Returns
    -------
    rec : dict
        Updated recording with filtered 'amp' data.
        
    Notes
    -----
    Pads data using 'edge' mode (repeats boundary values) to handle start/end points.
    Applies xr.DataArray.rolling().reduce(np.median) for computation.
    """
    pad_width = 1  # Adjust based on the kernel size
    padded_amp = rec['amp'].pad(time=(pad_width, pad_width), mode='edge')
    # Apply the median filter to the padded data
    filtered_padded_amp = padded_amp.rolling(time=median_filt, center=True).reduce(np.median)
    # Trim the padding after applying the filter
    rec['amp'] = filtered_padded_amp.isel(time=slice(pad_width, -pad_width))
    return rec    

#%% GLM FUNCTIONALITY 

def GLM(runs, cfg_GLM, geo3d, pruned_chans_list, stim_list):
    """
    Fit General Linear Model to estimate hemodynamic response functions.
    
    Performs GLM analysis on fNIRS concentration data across one or more runs.
    Constructs design matrix with HRF regressors, drift terms, and optional
    short-separation regressors. Supports OLS and AR-IRLS noise models.
    
    Parameters
    ----------
    runs : list of xr.DataArray
        List of concentration time series, one per run. Each with dimensions
        (channel, chromo, time).
    cfg_GLM : dict
        Configuration dictionary containing:
        - 't_pre', 't_post': Time window around events (cedalion.Quantity)
        - 't_delta', 't_std': Gaussian basis function parameters (cedalion.Quantity)
        - 'noise_model': 'ols' or 'ar_irls'
        - 'do_drift': bool, include polynomial drift regressors
        - 'do_drift_legendre': bool, include Legendre polynomial drift
        - 'drift_order': int, polynomial order for drift
        - 'do_short_sep': bool, include short-separation regressors
        - 'distance_threshold': cedalion.Quantity, threshold for short channels
        - 'short_channel_method': str, method for combining short channels
    geo3d : xr.DataArray
        3D optode geometry for identifying short-separation channels.
    pruned_chans_list : list of xr.DataArray
        List of channel quality scores, one per run, for masking short channels.
    stim_list : list of pd.DataFrame
        List of stimulus DataFrames, one per run, with columns:
        ['onset', 'duration', 'trial_type']
    
    Returns
    -------
    results : GLMResults
        Fitted GLM results from cedalion.models.glm.fit containing
        regression coefficients and statistics.
    hrf_estimate : xr.DataArray
        Estimated HRF with dimensions (channel, chromo, time, trial_type).
        Time coordinate is relative to event onset.
    hrf_mse : xr.DataArray
        Mean squared error of HRF estimate with dimensions
        (channel, chromo, time, trial_type). Quantifies uncertainty.
        
    Notes
    -----
    Multiple runs are concatenated along time dimension before fitting.
    Drift and short-separation regressors are computed per-run then combined.
    HRF is estimated by projecting beta coefficients onto Gaussian basis.
    """

    # 1. need to concatenate runs 
    if len(runs) > 1:
        Y_all, stim_df, runs_updated = concatenate_runs(runs, stim_list)
    else:
        Y_all = runs[0]
        stim_df = stim_list[0]
        runs_updated = runs
        
    run_unit = Y_all.pint.units
    # 2. define design matrix
    dms = glm.design_matrix.hrf_regressors(
                                    Y_all,
                                    stim_df,
                                    glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])
                                )

    # Combine drift and short-separation regressors (if any)
    if cfg_GLM['do_drift']:
        drift_regressors = get_drift_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_drift_legendre']:
        drift_regressors = get_drift_legendre_regressors(runs_updated, cfg_GLM)
        dms &= reduce(operator.and_, drift_regressors)

    if cfg_GLM['do_short_sep']:
        ss_regressors = get_short_regressors(runs_updated, pruned_chans_list, geo3d, cfg_GLM)
        dms &= reduce(operator.and_, ss_regressors)

    dms.common = dms.common.fillna(0)

    # 3. get betas and covariance
    results = glm.fit(Y_all, dms, noise_model=cfg_GLM['noise_model']) 
    betas = results.sm.params
    cov_params = results.sm.cov_params()

    # 4. estimate HRF and MSE
    basis_hrf = glm.GaussianKernels(cfg_GLM['t_pre'], cfg_GLM['t_post'], cfg_GLM['t_delta'], cfg_GLM['t_std'])(Y_all)

    trial_type_list = stim_df['trial_type'].unique()

    hrf_mse_list = []
    hrf_estimate_list = []

    for trial_type in trial_type_list:
        
        betas_hrf = betas.sel(regressor=betas.regressor.str.startswith(f"HRF {trial_type}"))
        hrf_estimate = estimate_HRF_from_beta(betas_hrf, basis_hrf)
        
        cov_hrf = cov_params.sel(regressor_r=cov_params.regressor_r.str.startswith(f"HRF {trial_type}"),
                            regressor_c=cov_params.regressor_c.str.startswith(f"HRF {trial_type}") 
                                    )
        hrf_mse = estimate_HRF_cov(cov_hrf, basis_hrf)

        hrf_estimate = hrf_estimate.expand_dims({'trial_type': [ trial_type ] })
        hrf_mse = hrf_mse.expand_dims({'trial_type': [ trial_type ] })

        hrf_estimate_list.append(hrf_estimate)
        hrf_mse_list.append(hrf_mse)

    hrf_estimate = xr.concat(hrf_estimate_list, dim='trial_type')
    hrf_estimate = hrf_estimate.pint.quantify(run_unit)

    hrf_mse = xr.concat(hrf_mse_list, dim='trial_type')
    hrf_mse = hrf_mse.pint.quantify(run_unit**2)

    # set universal time so that all hrfs have the same time base 
    fs = frequency.sampling_rate(runs[0]).to('Hz')
    before_samples = int(np.ceil((cfg_GLM['t_pre'] * fs).magnitude))
    after_samples = int(np.ceil((cfg_GLM['t_post'] * fs).magnitude))

    dT = np.round(1 / fs, 3)  # millisecond precision
    n_timepoints = len(hrf_estimate.time)
    reltime = np.linspace(-before_samples * dT, after_samples * dT, n_timepoints)

    hrf_mse = hrf_mse.assign_coords({'time': reltime})
    hrf_mse.time.attrs['units'] = 'second'

    hrf_estimate = hrf_estimate.assign_coords({'time': reltime})
    hrf_estimate.time.attrs['units'] = 'second'

    return results, hrf_estimate, hrf_mse


def estimate_HRF_cov(cov, basis_hrf):
    """
    Compute covariance (uncertainty) of HRF estimate from beta covariance.
    
    Projects the covariance matrix of GLM regression coefficients onto the
    Gaussian basis functions to obtain the time-varying uncertainty of the
    HRF estimate: Var(HRF) = B^T * Cov(beta) * B where B is the basis matrix.
    
    Parameters
    ----------
    cov : xr.DataArray
        Covariance matrix of beta coefficients with dimensions
        (channel, chromo, regressor_r, regressor_c) where regressor dimensions
        correspond to basis function coefficients.
    basis_hrf : xr.DataArray
        Gaussian basis functions with dimensions (component, channel, chromo, time)
        defining the temporal shape of HRF regressors.
    
    Returns
    -------
    mse_t : xr.DataArray
        Time-varying mean squared error (diagonal of HRF covariance) with
        dimensions (channel, chromo, time).
        
    Notes
    -----
    Computation: B^T @ Cov @ B using xr.dot for dimension alignment.
    Returns only diagonal (variance) not full covariance matrix.
    """

    basis_hrf = basis_hrf.rename({'component':'regressor_c'})
    basis_hrf = basis_hrf.assign_coords(regressor_c=cov.regressor_c.values)

    tmp = xr.dot(cov, basis_hrf, dims='regressor_c')

    tmp = tmp.rename({'regressor_r':'regressor'})
    basis_hrf = basis_hrf.rename({'regressor_c':'regressor'})

    mse_t = xr.dot(basis_hrf, tmp, dims='regressor')

    return mse_t

def estimate_HRF_from_beta(betas, basis_hrf):
    """
    Reconstruct hemodynamic response function from GLM beta coefficients.
    
    Projects beta weights onto Gaussian basis functions to obtain the time-varying
    HRF estimate: HRF(t) = sum_i beta_i * basis_i(t). Applies baseline correction
    by subtracting pre-stimulus mean.
    
    Parameters
    ----------
    betas : xr.DataArray
        GLM beta coefficients for HRF regressors with dimensions
        (channel, chromo, regressor) where regressors correspond to basis functions.
    basis_hrf : xr.DataArray
        Gaussian basis functions with dimensions (component, channel, chromo, time).
    
    Returns
    -------
    hrf_estimates_blcorr : xr.DataArray
        Baseline-corrected HRF estimate with dimensions (channel, chromo, time).
        Pre-stimulus period (time < 0) is subtracted to zero baseline.
        
    Notes
    -----
    Uses xr.dot to compute weighted sum of basis functions.
    Baseline correction removes pre-stimulus drift for cleaner HRF visualization.
    """
        
    basis_hrf = basis_hrf.rename({'component':'regressor'})
    basis_hrf = basis_hrf.assign_coords(regressor=betas.regressor.values)

    hrf_estimate = xr.dot(betas, basis_hrf, dims='regressor')

    hrf_estimates_blcorr = hrf_estimate - hrf_estimate.sel(time = hrf_estimate.time[hrf_estimate.time<0]).mean('time')

    return hrf_estimates_blcorr

def get_drift_regressors(runs, cfg_GLM):
    """
    Construct polynomial drift regressors for each run.
    
    Creates polynomial basis functions to model slow temporal drifts in the signal.
    Used with OLS noise model to account for baseline fluctuations. Each run gets
    independent drift regressors.
    
    Parameters
    ----------
    runs : list of xr.DataArray
        List of time series data, one per run.
    cfg_GLM : dict
        Configuration containing 'drift_order' (int): polynomial degree.
    
    Returns
    -------
    drift_regressors : list of DesignMatrix
        List of design matrices with drift regressors, one per run.
        Regressors are labeled 'Drift {order} run {i}'.
        
    Notes
    -----
    Uses cedalion.models.glm.design_matrix.drift_regressors.
    Typically used with OLS noise model. For AR-IRLS, use Legendre drift instead.
    """
    
    drift_regressors = []
    i=0
    for i, run  in enumerate(runs):
        drift = glm.design_matrix.drift_regressors(run, cfg_GLM['drift_order'])
        drift.common = drift.common.assign_coords({'regressor': [f'Drift {x} run {i}' for x in range(cfg_GLM['drift_order']+1)]})
        drift_regressors.append(drift)
        
    return drift_regressors

def get_drift_legendre_regressors(runs, cfg_GLM):
    """
    Construct Legendre polynomial drift regressors for each run.
    
    Creates orthogonal Legendre polynomial basis functions to model slow drifts.
    Preferred for AR-IRLS noise model due to better numerical properties.
    Each run gets independent drift regressors.
    
    Parameters
    ----------
    runs : list of xr.DataArray
        List of time series data, one per run.
    cfg_GLM : dict
        Configuration containing 'drift_order' (int): polynomial degree.
    
    Returns
    -------
    drift_regressors : list of DesignMatrix
        List of design matrices with Legendre drift regressors, one per run.
        Regressors are labeled 'Drift {order} run {i}'.
        
    Notes
    -----
    Uses cedalion.models.glm.design_matrix.drift_legendre_regressors.
    Legendre polynomials are orthogonal, improving GLM stability.
    Recommended for AR-IRLS noise model.
    """

    drift_regressors = []
    i=0
    for i, run  in enumerate(runs):

        drift = glm.design_matrix.drift_legendre_regressors(run, cfg_GLM['drift_order'])
        drift.common = drift.common.assign_coords({'regressor': [f'Drift {x} run {i}' for x in range(cfg_GLM['drift_order']+1)]})
        drift_regressors.append(drift)

    return drift_regressors

def get_short_regressors(runs, pruned_chans_list, geo3d, cfg_GLM):
    """
    Construct short-separation channel regressors for each run.
    
    Creates regressors from short-separation channels to model systemic physiological
    noise (cardiac, respiratory, blood pressure) that affects both short and long
    channels. Helps remove superficial scalp signals from brain signals.
    
    Parameters
    ----------
    runs : list of xr.DataArray
        List of concentration time series, one per run.
    pruned_chans_list : list of xr.DataArray
        List of channel quality scores for masking poor channels in short-channel
        identification.
    geo3d : xr.DataArray
        3D optode geometry for computing source-detector distances.
    cfg_GLM : dict
        Configuration containing:
        - 'distance_threshold': cedalion.Quantity, max distance for short channels (e.g., 20 mm)
        - 'short_channel_method': str, aggregation method ('mean', 'pca', etc.)
    
    Returns
    -------
    ss_regressors : list of DesignMatrix
        List of design matrices with short-separation regressors, one per run.
        Each labeled 'short run {i}'.
        
    Notes
    -----
    Short channels (SD distance < threshold) primarily measure scalp hemodynamics.
    Including these as regressors helps isolate brain signal in long channels.
    Poor quality channels are masked before computing short-channel average.
    """
    ss_regressors = []
    i=0
    for run, pruned_chans in zip(runs, pruned_chans_list):

        rec_pruned = prune_mask_ts(run, pruned_chans) # !!! how is this affected when using pruned data
        _, ts_short = nirs.split_long_short_channels(
                                rec_pruned, geo3d, distance_threshold= cfg_GLM['distance_threshold']  # !!! change to rec_pruned once NaN prob fixed
                                )

        short = glm.design_matrix.average_short_channel_regressor(ts_short)
        short.common = short.common.reset_coords('samples', drop=True)
        short.common = short.common.assign_coords({'regressor': [f'short run {i}']})
        ss_regressors.append(short)
        i = i+1

    return ss_regressors

def concatenate_runs(runs, stim):
    """
    Concatenate multiple runs along time dimension for joint analysis.
    
    Combines time series and stimulus timing from multiple runs into single
    continuous arrays. Adjusts time coordinates and stimulus onsets to maintain
    temporal continuity. Enables fitting a single GLM across all runs.
    
    Parameters
    ----------
    runs : list of xr.DataArray
        List of concentration time series, one per run, with dimensions
        (channel, chromo, time).
    stim : list of pd.DataFrame
        List of stimulus DataFrames, one per run, with columns:
        ['onset', 'duration', 'trial_type']
    
    Returns
    -------
    Y_all : xr.DataArray
        Concatenated time series with dimensions (channel, chromo, time).
        Time coordinates adjusted to be continuous across runs.
    stim_df : pd.DataFrame
        Concatenated stimulus DataFrame with adjusted onset times.
    runs_updated : list of xr.DataArray
        List of runs with updated time coordinates (for design matrix construction).
        
    Notes
    -----
    Time offset for each run is computed as: offset_i = last_time_{i-1} + dt
    All runs are converted to 'molar' units before concatenation.
    Maintains sampling rate continuity between runs.
    """

    CURRENT_OFFSET = 0
    runs_updated = []
    stim_updated = []

    for s, ts in zip(stim, runs):
        time = ts.time.values
        new_time = time + CURRENT_OFFSET

        ts_new = ts.copy(deep=True)
        ts_new = ts_new.pint.dequantify().pint.quantify('molar')
        ts_new = ts_new.assign_coords(time=new_time)

        stim_shift = s.copy()
        stim_shift['onset'] += CURRENT_OFFSET

        stim_updated.append(stim_shift)
        runs_updated.append(ts_new)

        CURRENT_OFFSET = new_time[-1] + (time[1] - time[0])

    Y_all = xr.concat(runs_updated, dim='time')
    Y_all.time.attrs['units'] = units.s
    stim_df = pd.concat(stim_updated, ignore_index = True)

    return Y_all, stim_df, runs_updated

#%% SPATIAL SMOOTHING 

def get_spatial_smoothing_kernel(V_ras, sigma_mm=80*units.mm):
    """
    Construct Gaussian spatial smoothing kernel for surface mesh.
    
    Creates a normalized Gaussian weight matrix where each row represents smoothing
    weights for one vertex based on Euclidean distances to all other vertices.
    Used to spatially smooth reconstructed images on brain/scalp surfaces.
    
    Parameters
    ----------
    V_ras : numpy.ndarray
        Vertex coordinates with shape (n_vertices, 3) in RAS orientation (mm).
    sigma_mm : cedalion.Quantity, optional
        Standard deviation of Gaussian kernel in mm (default: 80 mm).
        Controls smoothness scale.
    
    Returns
    -------
    W : numpy.ndarray
        Smoothing kernel matrix with shape (n_vertices, n_vertices) where
        W[i,j] is the weight for vertex j when smoothing vertex i.
        Each row sums to 1.0 (normalized). Weights below 1e-3 are set to 0.
        
    Notes
    -----
    Computation uses memory-efficient in-place operations:
    - W[i,j] = exp(-D[i,j]^2 / sigma^2) / row_sum[i]
    - Small weights (<1e-3) are zeroed for sparsity
    - Handles zero row sums (isolated vertices) by setting to 1.0
    
    Uses float32 precision to reduce memory usage for large meshes.
    """

    # distance matrix. float32 cuts memory in half vs float64
    D = cdist(V_ras.astype(np.float32), V_ras.astype(np.float32))  # shape (N, N), distances in mm

    # memory friendly in-place operations for calculating the gausisian weights
    W = D.astype(np.float32, copy=True)   # reuse same shape
    np.square(W, out=W)                   # W = D^2
    W /= (sigma_mm ** 2)                  # W = D^2 / sigma^2
    np.negative(W, out=W)                 # W = - D^2 / sigma^2
    np.exp(W, out=W)                      # W = exp(...)
    # set all columns of W to zero where log10(Adot_sum) < -2
    # set all rows of W to zero where log10(Adot_sum) < -2
    # set the diagonal to 1 in the rows where log10(Adot_sum) < -2
    W[W<1e-3] = 0

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    W /= row_sums

    return W
    
def compute_Hglobal_from_PCA(data_xr, variances, W):
    """
    Apply PCA-based spatial smoothing to reconstructed fNIRS images.
    
    Performs singular value decomposition (SVD) on mean-centered image time series,
    smooths the spatial principal components using a Gaussian kernel, then reconstructs
    the smoothed images. This preserves temporal dynamics while reducing spatial noise.
    
    Algorithm:
    1. For each (trial_type, subject, chromophore) slice:
       a. Extract image time series H as (time × vertex) matrix
       b. Mean-center: H_centered = H - mean(H, axis=time)
       c. SVD: H_centered = U @ diag(S) @ V^T
       d. Smooth spatial PCs: V* = W @ V (equivalently V*^T = V^T @ W^T)
       e. Reconstruct: H_smoothed = U @ diag(S) @ V*^T + mean
    
    Parameters
    ----------
    data_xr : xr.DataArray
        Reconstructed images with dimensions:
        - Required: ('subj', 'chromo', 'vertex', 'time' or 'reltime')
        - Optional: 'trial_type' (for event-related analyses)
    variances : xr.DataArray
        Variance estimates with matching dimensions (currently unused but kept
        for future variance-weighted smoothing).
    W : numpy.ndarray
        Gaussian smoothing kernel with shape (n_vertices, n_vertices) where
        W[i,j] is weight of vertex j in smoothing vertex i. Typically from
        get_spatial_smoothing_kernel().
    
    Returns
    -------
    Hglob : xr.DataArray
        Spatially smoothed images with same shape, coordinates, and units as input.
        Dimension order is preserved: ['trial_type'(if present), 'subj', 'chromo',
        'vertex', 'time'/'reltime'].
        
    Notes
    -----
    - Uses economy SVD (full_matrices=False) for efficiency
    - Processes each slice independently to reduce memory usage
    - Normalizes smoothing kernel rows to sum to 1.0 before application
    - Handles units by temporarily removing them during computation
    - Time dimension can be named 'time' or 'reltime'
    
    Memory optimization:
    - Uses float32 for intermediate computations
    - Pre-allocates output array
    - Avoids creating full covariance matrices
    """
    # Figure out time dimension name
    time_dim = 'reltime' if 'reltime' in data_xr.dims else 'time'

    # Drop units for speed and to avoid pint assignment issues
    has_units = hasattr(data_xr, 'pint') and getattr(data_xr.pint, 'units', None) is not None
    units = data_xr.pint.units if has_units else None
    dx = data_xr.pint.dequantify() if has_units else data_xr

    # Output container (unitless for now)
    Hglob = dx.copy(deep=True)

    W_orig = W.copy()
    
    # Build label->index maps for fast .data assignment
    tt_idx = {v: i for i, v in enumerate(dx.trial_type.values)} if "trial_type" in dx.dims else None
    sj_idx = {v:i for i,v in enumerate(dx.subj.values)}
    ch_idx = {v:i for i,v in enumerate(dx.chromo.values)}

    trial_types = dx.trial_type.values if "trial_type" in dx.dims else [None]

    # Iterate over slices
    for tt in trial_types:
        for sj in dx.subj.values:
            for ch in dx.chromo.values:
                # Extract H slice as (time × vertex)
                sel_dict = dict(subj=sj, chromo=ch)
                if tt is not None:
                    sel_dict["trial_type"] = tt

                H = dx.sel(sel_dict).transpose(time_dim, 'vertex').values.astype(np.float32, copy=False)
                var = variances.sel(sel_dict).values

                W = W_orig
                row_sums = W.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                W /= row_sums

                WT = W.T.astype(np.float32, copy=False)

                mean = H.mean(axis=1, keepdims=True)
                H_centered = H - mean
                
                # Economy SVD: H = U @ diag(S) @ V^T
                U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
                
                # Smooth spatial PCs: V*^T = V^T @ W^T
                Vt_star = Vt @ WT
                
                # Reconstruct smoothed image
                Hglob_slice = U @ np.diag(S) @ Vt_star
                Hglob_slice_recon = Hglob_slice + mean
                
                # Write back as (vertex × time)
                if tt is not None:
                    Hglob.data[tt_idx[tt], sj_idx[sj], ch_idx[ch], :, :] = Hglob_slice_recon.T
                else:
                    Hglob.data[sj_idx[sj], ch_idx[ch], :, :] = Hglob_slice_recon.T
    
    # Reattach units if present
    if has_units:
        Hglob = Hglob.pint.quantify(units)

    final_dims = ['trial_type'] if 'trial_type' in dx.dims else []
    final_dims += ['subj', 'chromo', 'vertex', time_dim]

    return Hglob.transpose(*final_dims)
