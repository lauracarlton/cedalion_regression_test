#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_recon_func.py

fNIRS image reconstruction utilities. This module provides functions for performing
regularized inverse problems to reconstruct spatial maps of hemodynamic changes from
fNIRS measurements. Supports both direct (chromophore space) and indirect (wavelength
space) reconstruction methods with flexible regularization schemes.

Key Functionality:
- Forward model setup: Load head models, probe geometries, and sensitivity matrices
- Regularization: Spatial and measurement regularization with automatic parameter scaling
- Inverse matrix computation: Calculate reconstruction operators (W matrices)
- Image reconstruction: Transform measurements to images in brain/scalp space
- Uncertainty quantification: Compute posterior variance of reconstructed images
- Probe registration: Align probe coordinates to head model coordinate systems

Reconstruction Methods:
- Direct: Solves for HbO/HbR concentrations directly using stacked forward model
- Indirect: Solves wavelength-by-wavelength, then converts OD to concentrations

Regularization:
- Spatial: Column-scaling based on forward model sensitivity
- Measurement: Ridge regression in data space with noise covariance
- Automatic parameter scaling: lambda_R ensures consistency between methods

Author: Laura Carlton | lcarlton@bu.edu
"""

import os.path
import sys

import numpy as np
import xarray as xr

import cedalion
import cedalion.dataclasses as cdc
import cedalion.dataclasses.geometry as geo
import cedalion.datasets as datasets
import cedalion.geometry.landmarks as cgeolm
import cedalion.imagereco.forward_model as fw
import cedalion.io as io
import cedalion.nirs as nirs
import cedalion.xrutils as xrutils
from cedalion import units
from cedalion.io.forward_model import load_Adot

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import spatial_basis_func as sbf


#%% DATA LOADING

def load_head_model(head_model='ICBM152', with_parcels=True):
    """
    Load anatomical head model with brain and scalp surfaces.
    
    Loads segmentation masks, landmarks, and surface meshes for a standard
    head model. Optionally includes parcellation labels for region-of-interest
    analysis.
    
    Parameters
    ----------
    head_model : str, optional
        Head model to load: 'ICBM152' or 'colin27' (default: 'ICBM152').
    with_parcels : bool, optional
        Whether to include parcellation labels (default: True).
    
    Returns
    -------
    head : fw.TwoSurfaceHeadModel
        Two-layer head model with brain and scalp surfaces, landmarks,
        and optional parcel labels. Surfaces have units of mm.
    PARCEL_DIR : str or None
        Path to parcellation file if with_parcels=True, else None.
        
    Notes
    -----
    Uses cedalion.datasets to retrieve paths to segmentation data.
    Surfaces are smoothed (smoothing=0.5) and holes are filled during construction.
    Scalp and brain surfaces are in RAS (Right-Anterior-Superior) coordinates.
    """
    
    if head_model == 'ICBM152':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_icbm152_segmentation()
        if with_parcels:
            PARCEL_DIR = datasets.get_icbm152_parcel_file()
        else :
            PARCEL_DIR = None
            
    elif head_model == 'colin27':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_colin27_segmentation()
        if with_parcels:
            PARCEL_DIR = datasets.get_colin27_parcel_file()
        else :
            PARCEL_DIR = None
            
    masks, t_ijk2ras = io.read_segmentation_masks(SEG_DATADIR, mask_files)

    
    head = fw.TwoSurfaceHeadModel.from_surfaces(
        segmentation_dir=SEG_DATADIR,
        mask_files = mask_files,
        brain_surface_file= os.path.join(SEG_DATADIR, "mask_brain.obj"),
        scalp_surface_file= os.path.join(SEG_DATADIR, "mask_scalp.obj"),
        landmarks_ras_file=landmarks_file,
        smoothing=0.5,
        fill_holes=True,
        parcel_file=PARCEL_DIR
    ) 
    head.scalp.units = units.mm
    head.brain.units = units.mm
    
    return head, PARCEL_DIR


def load_probe(probe_path, snirf_name ='fullhead_56x144_System2.snirf', head_model='ICBM152'):
    """
    Load forward model sensitivity matrix and probe geometry.
    
    Reads pre-computed forward model (Adot) and probe optode positions from
    stored files. The forward model maps vertex activations to channel measurements.
    
    Parameters
    ----------
    probe_path : str
        Root directory containing 'fw' subdirectory with forward model files.
    snirf_name : str, optional
        Name of SNIRF file containing probe geometry (default: 'fullhead_56x144_System2.snirf').
    head_model : str, optional
        Head model name matching subdirectory in probe_path/fw/ (default: 'ICBM152').
    
    Returns
    -------
    Adot : xr.DataArray
        Forward model sensitivity matrix with dimensions (channel, vertex, wavelength).
        Includes coordinates 'is_brain' and optionally 'parcel'.
    meas_list : list
        Measurement list from SNIRF file with source-detector pairings.
    geo3d : xr.DataArray
        3D optode geometry with source/detector positions.
    amp : xr.DataArray
        Template amplitude array with channel structure.
        
    Notes
    -----
    Expects forward model stored as 'Adot.nc' in probe_path/fw/head_model/.
    SNIRF file provides probe geometry and channel definitions.
    """
        
    Adot = load_Adot(os.path.join(probe_path, 'fw',  head_model, 'Adot.nc'))

    recordings = io.read_snirf(os.path.join(probe_path, 'fw',  head_model, snirf_name))
    rec = recordings[0]
    geo3d = rec.geo3d
    amp = rec['amp']
    meas_list = rec._measurement_lists['amp']

    return Adot, meas_list, geo3d, amp


#%% MATRIX CALCULATIONS

def compute_lambda_R_indirect(Adot,
    lambda_R,
    alpha_spatial,
    wavelengths,
):
    """
    Compute wavelength-specific prior scaling parameter for indirect reconstruction.
    
    Scales lambda_R to ensure consistency between direct
    (chromophore space) and indirect (wavelength space) methods. Uses extinction
    coefficients to relate chromophore regularization strength to OD regularization.
    
    Parameters
    ----------
    Adot : xr.DataArray
        Forward model with dimensions (channel, vertex, wavelength).
    lambda_R : float
        scaling parameter for direct method.
    alpha_spatial : float
        Spatial regularization parameter.
    wavelengths : array-like
        Wavelength values to compute parameters for.
    
    Returns
    -------
    lambda_R_indirect : xr.DataArray
        Wavelength-specific parameter with dimension (wavelength,).
        Scaled to match direct method's effective regularization strength.
        
    Notes
    -----
    Algorithm:
    1. Compute direct method spatial prior R_direct
    2. Extract maximum R values for HbO brain vertices
    3. Convert to OD space: R_OD = E^2 @ R_chromophore
    4. Scale each wavelength's lambda to match converted values
    
    Ensures reconstructions from direct and indirect methods have comparable
    spatial smoothness and noise suppression.
    """

    ec  = nirs.get_extinction_coefficients('prahl', wavelengths)

    A_stacked = get_Adot_scaled(Adot, wavelengths)
    nV_brain = Adot.is_brain.sum().values
    nV_head = Adot.shape[1]

    R_direct = _calculate_prior_R(A_stacked, alpha_spatial=alpha_spatial)
    R_direct = R_direct * lambda_R

    R_direct_max = [R_direct[:nV_brain].max().values, R_direct[nV_head:nV_head+nV_brain].max().values]

    # Convert direct prior to indirect (OD space)
    R_indirect_wl1 = _calculate_prior_R(Adot.isel(wavelength=0), alpha_spatial=alpha_spatial)
    R_indirect_wl2 = _calculate_prior_R(Adot.isel(wavelength=1), alpha_spatial=alpha_spatial)
    R_indirect_converted = ec.values**2 @ R_direct_max

    lambda_wl1 = R_indirect_converted[0] / R_indirect_wl1[:nV_brain].max()
    lambda_wl2 = R_indirect_converted[1] / R_indirect_wl2[:nV_brain].max()

    lambda_R_indirect = xr.DataArray([lambda_wl1, lambda_wl2], 
                                dims=['wavelength'],
                                coords={'wavelength': wavelengths})

    return lambda_R_indirect
    
def get_Adot_scaled(Adot, wavelengths):
    """
    Stack forward model for direct chromophore reconstruction.
    
    Multiplies forward model by extinction coefficients and stacks wavelengths
    to create a single matrix mapping chromophore concentrations to measurements:
    A_stacked = [E(λ1) * Adot(λ1); E(λ2) * Adot(λ2)] for both HbO and HbR.
    
    Parameters
    ----------
    Adot : xr.DataArray
        Forward model with dimensions (channel, vertex, wavelength).
    wavelengths : array-like
        Two wavelength values (typically [760nm, 850nm]).
    
    Returns
    -------
    A : xr.DataArray
        Stacked forward model with dimensions (measurement, flat_vertex) where:
        - measurement = 2 * n_channels (both wavelengths)
        - flat_vertex = 2 * n_vertices (HbO then HbR for each tissue layer)
        Layout: [[E_HbO(λ1)*A(λ1), E_HbR(λ1)*A(λ1)],
                 [E_HbO(λ2)*A(λ2), E_HbR(λ2)*A(λ2)]]
        Inherits 'parcel' and 'is_brain' coordinates if present in Adot.
        
    Notes
    -----
    Used for direct method where inverse problem is solved in chromophore space.
    Extinction coefficients convert absorption changes to concentration changes.
    """
    
    nchannel = Adot.shape[0]
    nvertices = Adot.shape[1]
    E = nirs.get_extinction_coefficients('prahl', Adot.wavelength)

    A = np.zeros((2 * nchannel, 2 * nvertices))
    wl1 = wavelengths[0]
    wl2 = wavelengths[1]
    
    A[:nchannel, :nvertices] = E.sel(chromo="HbO", wavelength=wl1).values * Adot.sel(wavelength=wl1)
    A[:nchannel, nvertices:] = E.sel(chromo="HbR", wavelength=wl1).values * Adot.sel(wavelength=wl1)
    A[nchannel:, :nvertices] = E.sel(chromo="HbO", wavelength=wl2).values * Adot.sel(wavelength=wl2)
    A[nchannel:, nvertices:] = E.sel(chromo="HbR", wavelength=wl2).values * Adot.sel(wavelength=wl2)

    A = xr.DataArray(A, dims=("measurement", "flat_vertex"))

    if "parcel" in Adot.coords:
        A = A.assign_coords({"parcel" : ("flat_vertex", np.concatenate((Adot.coords['parcel'].values, Adot.coords['parcel'].values)))})
    if "is_brain" in Adot.coords:
        A = A.assign_coords({"is_brain" : ("flat_vertex", np.concatenate((Adot.coords['is_brain'].values, Adot.coords['is_brain'].values)))})
    return A

def calculate_W(A, 
                lambda_R=1e-6, 
                alpha_meas=1e3, 
                alpha_spatial=1e-3,
                DIRECT=True, 
                C_meas_flag=False, 
                C_meas=None, 
                D=None, 
                F=None, 
                max_eig=None):
    """
    Compute regularized inverse matrix (reconstruction operator).
    
    Calculates W matrix that maps measurements to reconstructed images using
    Tikhonov regularization with spatial and measurement regularization:
    W = D @ (F + λ_meas * C_meas)^(-1) where:
    - D = R * A^T (scaled adjoint)
    - F = A * R * A^T (regularized Gram matrix)
    - R = spatial prior (diagonal matrix)
    
    Parameters
    ----------
    A : xr.DataArray
        Forward model matrix. For DIRECT=True: stacked (measurement, flat_vertex).
        For DIRECT=False: unstacked (channel, vertex, wavelength).
    lambda_R : float, optional
        Ridge parameter scaling spatial regularization (default: 1e-6).
    alpha_meas : float, optional
        Measurement regularization weight (default: 1e3).
    alpha_spatial : float, optional
        Spatial regularization weight (default: 1e-3).
    DIRECT : bool, optional
        True for direct chromophore reconstruction, False for indirect (default: True).
    C_meas_flag : bool, optional
        Whether to use measurement covariance matrix (default: False).
    C_meas : array-like, optional
        Measurement covariance (diagonal elements). If None, uses identity.
    D, F, max_eig : optional
        Pre-computed intermediate matrices for repeated calculations. If None,
        computed from A.
    
    Returns
    -------
    W_xr : xr.DataArray
        Inverse matrix. For DIRECT: (flat_vertex, measurement).
        For INDIRECT: (wavelength, vertex, channel).
    D_xr : xr.DataArray
        Scaled adjoint matrix with same structure as W.
    F_xr : xr.DataArray
        Regularized Gram matrix for measurement space.
    max_eig : float or xr.DataArray
        Maximum eigenvalue of unscaled Gram matrix, used for lambda_meas scaling.
        
    Notes
    -----
    Spatial regularization uses column scaling: R = diag(1 / (B + λ_s)) where
    B = sum(A^2, axis=0).
    
    Measurement regularization: λ_meas = alpha_meas * lambda_R * max_eig
    ensures consistent scaling across parameter choices.
    """
    
    
    if DIRECT:
        if C_meas_flag:
            C_meas = np.diag(C_meas)

        W_xr, D, F, max_eig = _calculate_W_direct(A, 
                                                lambda_R=lambda_R, 
                                                alpha_meas=alpha_meas, 
                                                alpha_spatial=alpha_spatial, 
                                                C_meas_flag=C_meas_flag, 
                                                C_meas=C_meas,
                                                D=D, 
                                                F=F, 
                                                max_eig=max_eig)
    else:
        W_xr, D, F, max_eig = _calculate_W_indirect(A, 
                                                    lambda_R=lambda_R, 
                                                    alpha_meas=alpha_meas, 
                                                    alpha_spatial=alpha_spatial, 
                                                    C_meas_flag=C_meas_flag, 
                                                    C_meas=C_meas, 
                                                    D=D, 
                                                    F=F, 
                                                    max_eig=max_eig)
                    
    return W_xr, D, F, max_eig

def _calculate_prior_R(A, alpha_spatial):
    """
    Compute spatial regularization prior (column scaling matrix).
    
    Calculates diagonal regularization matrix based on forward model sensitivity:
    R_j = 1 / (sum_i A_ij^2 + λ_spatial) where λ_spatial is scaled by max sensitivity.
    Vertices with high sensitivity get less regularization; low sensitivity vertices
    are smoothed more heavily.
    
    Parameters
    ----------
    A : numpy.ndarray or xr.DataArray
        Forward model matrix with shape (n_channels, n_vertices) or similar.
    alpha_spatial : float
        Spatial regularization weight controlling smoothness strength.
    
    Returns
    -------
    R : numpy.ndarray or xr.DataArray
        Diagonal regularization matrix (as 1D array of diagonal elements)
        with same shape as columns of A.
        
    Notes
    -----
    Implements sensitivity-based column scaling:
    1. Compute B_j = sum_i A_ij^2 (total sensitivity at vertex j)
    2. Scale: λ_spatial = alpha_spatial * max(B)
    3. Regularize: R_j = 1 / (B_j + λ_spatial)
    
    Equivalent to diagonal of (A^T A + λI)^(-1) in limit of strong regularization.
    """

    B = np.sum((A ** 2), axis=0)
    b = B.max()
    
    lambda_spatial = alpha_spatial * b
    
    L = np.sqrt(B + lambda_spatial)
    Linv = 1/L
    R = Linv**2         

    return R

def _calculate_W_direct(A, 
                        alpha_spatial=1e-3, 
                        alpha_meas=1e4,
                        lambda_R=1e-6,
                        C_meas_flag=False, 
                        C_meas=None, 
                        D=None, 
                        F=None, 
                        max_eig=None):
    """
    Compute inverse matrix for direct chromophore reconstruction.
    
    Internal function implementing regularized inversion for stacked forward model
    in chromophore space. Calculates W = D @ (F + λ * C)^(-1) with spatial and
    measurement regularization.
    
    Parameters
    ----------
    A : xr.DataArray
        Stacked forward model with dimensions (measurement, flat_vertex).
    alpha_spatial : float, optional
        Spatial regularization weight (default: 1e-3).
    alpha_meas : float, optional
        Measurement regularization weight (default: 1e4).
    lambda_R : float, optional
        Ridge parameter for spatial prior (default: 1e-6).
    C_meas_flag : bool, optional
        Use measurement covariance (default: False).
    C_meas : numpy.ndarray, optional
        Measurement covariance matrix (2D) if C_meas_flag=True.
    D, F, max_eig : optional
        Pre-computed matrices. If None, computed from A.
    
    Returns
    -------
    W_xr : xr.DataArray
        Inverse matrix with dimensions (flat_vertex, measurement).
    D_xr : xr.DataArray
        Scaled adjoint: D = R * A^T.
    F_xr : xr.DataArray
        Gram matrix: F = A * R * A^T with dimensions (measurement1, measurement2).
    max_eig : float
        Maximum eigenvalue of unscaled Gram matrix for lambda_meas scaling.
        
    Notes
    -----
    Computation steps:
    1. Spatial prior: R = diag(1/(B + λ_s)) * lambda_R
    2. Scaled forward model: AR = A * R
    3. Gram matrix: F = AR @ A^T
    4. Max eigenvalue: max_eig from unscaled version
    5. Measurement regularization: λ_meas = alpha_meas * lambda_R * max_eig
    6. Inverse: W = R * A^T @ (F + λ_meas * C)^(-1)
    """
    
    A_coords = A.coords
    A = A.pint.dequantify().values
                     
    if D is None and F is None:
        nV_hbo = A.shape[1] // 2

        R = _calculate_prior_R(A, alpha_spatial)
        Linv = np.sqrt(R)
        R = R * lambda_R

        AR = A * R 
        F = AR @ A.T

        # Get F without R scaling to define max eigenvalue
        A_hat = A * Linv
        F_unscaled = A_hat @ A_hat.T 
        max_eig = np.max(np.linalg.eigvals(F_unscaled)) 

        D = R[:, np.newaxis] * A.T

    else:
        D = D.values
        F = F.values
        
    lambda_meas = alpha_meas * max_eig * lambda_R
    
    if C_meas_flag:  
        assert  len(C_meas.shape) == 2                 
        W = D @ np.linalg.inv(F  + lambda_meas * C_meas )
    else:
        W = D @ np.linalg.inv(F  + lambda_meas * np.eye(A.shape[0]) )
    
    W_xr = xr.DataArray(W, dims=("flat_vertex", "measurement"))
    D_xr = xr.DataArray(D, dims=("flat_vertex", "measurement"))

    if 'parcel' in A_coords:
        W_xr = W_xr.assign_coords({"parcel" : ("flat_vertex", A_coords['parcel'].values)})
        D_xr = D_xr.assign_coords({"parcel" : ("flat_vertex", A_coords['parcel'].values)})
    if 'is_brain' in A_coords:
        W_xr = W_xr.assign_coords({"is_brain": ("flat_vertex", A_coords['is_brain'].values)}) 
        D_xr = D_xr.assign_coords({"is_brain": ("flat_vertex", A_coords['is_brain'].values)})
    
    F_xr = xr.DataArray(F, dims=("measurement1", "measurement2"))

    return W_xr, D_xr, F_xr, max_eig

def _calculate_W_indirect(A, 
                        lambda_R=1e-6, 
                        alpha_meas=1e3, 
                        alpha_spatial=1e-3, 
                        C_meas_flag=False, 
                        C_meas=None, 
                        D=None, 
                        F=None, 
                        max_eig=None):
    """
    Compute inverse matrices for indirect wavelength-by-wavelength reconstruction.
    
    Internal function implementing regularized inversion separately for each wavelength.
    Applies wavelength-specific lambda_R scaling to ensure consistency with direct method.
    
    Parameters
    ----------
    A : xr.DataArray
        Forward model with dimensions (channel, vertex, wavelength).
    lambda_R : float, optional
        Base ridge parameter (default: 1e-6).
    alpha_meas : float, optional
        Measurement regularization weight (default: 1e3).
    alpha_spatial : float, optional
        Spatial regularization weight (default: 1e-3).
    C_meas_flag : bool, optional
        Use wavelength-specific measurement covariance (default: False).
    C_meas : xr.DataArray, optional
        Measurement covariance with wavelength dimension.
    D, F, max_eig : optional
        Pre-computed wavelength-specific matrices.
    
    Returns
    -------
    W_xr : xr.DataArray
        Inverse matrices with dimensions (wavelength, vertex, channel).
    D_xr : xr.DataArray
        Scaled adjoint matrices for each wavelength.
    F_xr : xr.DataArray
        Gram matrices with dimensions (wavelength, channel1, channel2).
    max_eig : xr.DataArray
        Maximum eigenvalues with dimension (wavelength,).
        
    Notes
    -----
    Calls compute_lambda_R_indirect() to get wavelength-specific ridge parameters
    that match the effective regularization of the direct method.
    
    Each wavelength solved independently:
    W(λ) = R(λ) * A(λ)^T @ (F(λ) + λ_meas(λ) * C(λ))^(-1)
    """
    
    lambda_R_indirect = compute_lambda_R_indirect(A, lambda_R, alpha_spatial, A.wavelength)

    W = []
    D_lst = []
    F_lst = []
    max_eig_lst = []

    for wavelength in A.wavelength:
        
        if C_meas_flag:
            C_meas_wl = C_meas.sel(wavelength=wavelength).values
            C_meas_wl = np.diag(C_meas_wl)
        else:
            C_meas_wl = None
            
        lambda_R_wl = lambda_R_indirect.sel(wavelength=wavelength).values

        if F is None and D is None:

            A_wl = A.sel(wavelength=wavelength).values

            R = _calculate_prior_R(A_wl, alpha_spatial)
            Linv = np.sqrt(R)

            R = R * lambda_R_wl

            A_tmp = A_wl * R
            F_wl = A_tmp @ A_wl.T

            A_hat = A_wl * Linv
            F_unscaled = A_hat @ A_hat.T

            D_wl = R[:, np.newaxis] * A_wl.T

            max_eig_wl = np.max(np.linalg.eigvals(F_unscaled)) 

        else:
            F_wl = F.sel(wavelength=wavelength).values
            D_wl = D.sel(wavelength=wavelength).values
            max_eig_wl = max_eig.sel(wavelength=wavelength).values

        lambda_meas = alpha_meas * lambda_R_wl * max_eig_wl

        W_wl = D_wl @ np.linalg.inv(F_wl  + lambda_meas * C_meas_wl )
        
        W.append(W_wl)
        D_lst.append(D_wl)
        F_lst.append(F_wl)
        max_eig_lst.append(max_eig_wl)

    W_xr = xr.DataArray(W, dims=( "wavelength", "vertex", "channel",),
                        coords = {'wavelength': A.wavelength})
    
    D_xr = xr.DataArray(D_lst, dims=( "wavelength", "vertex", "channel",),
                        coords = {'wavelength': A.wavelength})

    F_xr = xr.DataArray(F_lst, dims=( "wavelength", "channel1", "channel2"),
                        coords = {'wavelength': A.wavelength})

    max_eig = xr.DataArray(max_eig_lst, dims=("wavelength"),
                        coords = {'wavelength': A.wavelength})

    return W_xr, D_xr, F_xr, max_eig

#%% IMAGE RECONSTRUCTION

def _get_image_brain_scalp_direct(y, W, SB=False, G=None):
    """
    Reconstruct chromophore images using direct method.
    
    Applies inverse matrix to stacked measurements to obtain HbO and HbR
    concentrations directly. Handles optional spatial basis function representation.
    
    Parameters
    ----------
    y : xr.DataArray
        Measurements with dimensions (channel, wavelength) or (channel, wavelength, time).
    W : xr.DataArray
        Inverse matrix with dimensions (flat_vertex, measurement).
    SB : bool, optional
        Whether spatial basis functions were used (default: False).
    G : dict, optional
        Spatial basis matrices if SB=True. Contains 'G_brain' and 'G_scalp'.
    
    Returns
    -------
    X : xr.DataArray
        Reconstructed chromophore images with dimensions:
        - (vertex, chromo) for single timepoint
        - (vertex, time, chromo) for time series
        Chromophore coordinate: ['HbO', 'HbR'].
        
    Notes
    -----
    Stacks measurements as [λ1_ch1, λ1_ch2, ..., λ2_ch1, λ2_ch2, ...]
    Computes X = W @ y, then reshapes from flat [HbO_vertices; HbR_vertices]
    to structured (vertex, chromo) format.
    If SB=True, converts from kernel space to image space using G matrices.
    """
    
    y = y.stack(measurement=['channel', 'wavelength']).sortby('wavelength')

    if len(y.shape) > 1:
        y = y.transpose('measurement', 'time')

    X = W.values @ y.values

    split = len(X)//2
    
    if SB:
        X = sbf.go_from_kernel_space_to_image_space_direct(X, G)

    else:
        if len(X.shape) == 1:
            X = X.reshape([2, split]).T
        else:
            X = X.reshape([2, split, X.shape[1]])
            X = X.transpose(1,2,0)
    
    if len(X.shape) == 2:
        X = xr.DataArray(X, 
                         dims = ('vertex', 'chromo'),
                         coords = {'chromo': ['HbO', 'HbR']}
                         )
    else:
        if 'time' in y.dims:
            t = y.time
            t_name = 'time'
        elif 'reltime' in y.dims:
            t = y.reltime
            t_name = 'reltime'
        X = xr.DataArray(X, 
                         dims = ('vertex',  t_name, 'chromo',),
                         coords = {'chromo': ['HbO', 'HbR'],
                                   t_name: t},
                         )

    return X


def _get_image_brain_scalp_indirect(y, W, SB=False, G=None):
    """
    Reconstruct chromophore images using indirect method.
    
    Solves for optical density changes at each wavelength independently,
    then converts to chromophore concentrations using extinction coefficients.
    Handles optional spatial basis function representation.
    
    Parameters
    ----------
    y : xr.DataArray
        Measurements with dimensions (channel, wavelength, time) or (channel, wavelength).
    W : xr.DataArray
        Wavelength-specific inverse matrices with dimensions (wavelength, vertex, channel).
    SB : bool, optional
        Whether spatial basis functions were used (default: False).
    G : dict, optional
        Spatial basis matrices if SB=True.
    
    Returns
    -------
    X : xr.DataArray
        Reconstructed chromophore images with dimensions:
        - (vertex, chromo) for single timepoint
        - (vertex, chromo, time) for time series
        Chromophore coordinate: ['HbO', 'HbR'].
        
    Notes
    -----
    Algorithm:
    1. For each wavelength: X_OD(λ) = W(λ) @ y(λ)
    2. If SB=True: convert kernel space to image space
    3. Stack wavelengths: X_OD = [X_OD(λ1), X_OD(λ2)]
    4. Convert to concentrations: X_chromo = E^(-1) @ X_OD
    
    Uses Moore-Penrose pseudo-inverse of extinction coefficient matrix.
    """
                  
    W_indirect_wl0 = W.isel(wavelength=0)
    W_indirect_wl1 = W.isel(wavelength=1)
    
    if len(y.shape) > 2:
        y = y.transpose('channel', 'time', 'wavelength')

    y_wl0 = y.isel(wavelength=0) 
    y_wl1 = y.isel(wavelength=1)

    X_wl0 = W_indirect_wl0.values @ y_wl0.values
    X_wl1 = W_indirect_wl1.values @ y_wl1.values
             
    if SB:
        X_wl0 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl0, G)
        X_wl1 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl1, G)
    
    X_od = np.stack([X_wl0, X_wl1], axis=1)  

    if len(X_od.shape) == 2:
        X_od = xr.DataArray(X_od, 
                        dims = ('vertex', 'wavelength'),
                        coords = {'wavelength': W.wavelength}
                        )
    else:
        if 'time' in y.dims:
            t = y.time
            t_name = 'time'
        elif 'reltime' in y.dims:
            t = y.reltime
            t_name = 'reltime'
        X_od = xr.DataArray(X_od, 
                        dims = ('vertex', 'wavelength', t_name),
                        coords = {'wavelength': W.wavelength,
                                t_name: t},
                        )

    # convert to concentration 
    E = nirs.get_extinction_coefficients('prahl', W.wavelength)
    einv = xrutils.pinv(E)

    X = xr.dot(einv, X_od/units.mm, dims=["wavelength"])
    
    return X

def do_image_recon(od, 
                   head, 
                   Adot, 
                   C_meas_flag, 
                   C_meas, 
                   wavelength, 
                   DIRECT,
                   SB, 
                   cfg_sbf, 
                   lambda_R, 
                   alpha_spatial, 
                   alpha_meas, 
                   D, 
                   F, 
                   G, 
                   max_eig):
    """
    Main image reconstruction function with flexible configuration.
    
    Orchestrates the full reconstruction pipeline: forward model preparation,
    spatial basis function setup (optional), inverse matrix calculation, and
    image reconstruction. Supports both direct and indirect methods.
    
    Parameters
    ----------
    od : xr.DataArray
        Optical density measurements with dimensions (channel, wavelength, time)
        or (channel, wavelength).
    head : TwoSurfaceHeadModel
        Head model with brain and scalp surfaces.
    Adot : xr.DataArray
        Forward model with dimensions (channel, vertex, wavelength).
    C_meas_flag : bool
        Whether to use measurement covariance matrix.
    C_meas : xr.DataArray or None
        Measurement covariance if C_meas_flag=True.
    wavelength : array-like
        Wavelength values (typically 2 elements).
    DIRECT : bool
        True for direct chromophore reconstruction, False for indirect.
    SB : bool
        Whether to use spatial basis functions.
    cfg_sbf : dict
        Spatial basis function configuration containing:
        - 'mask_threshold': sensitivity threshold for mask
        - 'threshold_brain', 'threshold_scalp': downsampling distances
        - 'sigma_brain', 'sigma_scalp': Gaussian kernel widths
    lambda_R : float
        Ridge regularization parameter.
    alpha_spatial : float
        Spatial regularization weight.
    alpha_meas : float
        Measurement regularization weight.
    D, F, G, max_eig : optional
        Pre-computed matrices. If None, computed from scratch.
    
    Returns
    -------
    X : xr.DataArray
        Reconstructed images with coordinates 'is_brain' and optionally 'parcel'.
    W : xr.DataArray
        Inverse matrix used for reconstruction.
    D : xr.DataArray
        Scaled adjoint matrix.
    F : xr.DataArray
        Gram matrix in measurement space.
    G : dict or None
        Spatial basis matrices if SB=True.
    max_eig : float or xr.DataArray
        Maximum eigenvalue(s) for regularization scaling.
        
    Notes
    -----
    Automatically handles forward model transformation when spatial basis
    functions are enabled. Preserves coordinates from original forward model.
    """
    
    Adot_tmp = Adot.copy()
    if DIRECT:
        Adot_stacked = get_Adot_scaled(Adot, wavelength)
        
        if SB:
            if G is None:
                M = sbf.get_sensitivity_mask(Adot, cfg_sbf['mask_threshold'], 1)
                G = sbf.get_G_matrix(head, 
                                    M, 
                                    threshold_brain=cfg_sbf['threshold_brain'],
                                    threshold_scalp = cfg_sbf['threshold_scalp'],
                                    sigma_brain=cfg_sbf['sigma_brain'],
                                    sigma_scalp=cfg_sbf['sigma_scalp'])
                
            H_stacked = sbf.get_H_stacked(G, Adot_stacked)
            Adot_stacked = H_stacked.copy()
            
        W, D, F, max_eig = calculate_W(Adot_stacked, 
                                        lambda_R=lambda_R, 
                                        alpha_meas=alpha_meas, 
                                        alpha_spatial=alpha_spatial, 
                                        C_meas_flag=C_meas_flag, 
                                        C_meas=C_meas, 
                                        DIRECT=DIRECT, 
                                        D=D, 
                                        F=F, 
                                        max_eig=max_eig)
       
        X = _get_image_brain_scalp_direct(od, W, SB=SB, G=G)
    
            
    else:
        if SB:
            if G is None:
                M = sbf.get_sensitivity_mask(Adot, cfg_sbf['mask_threshold'], 1)
                G = sbf.get_G_matrix(head, 
                                    M, 
                                    threshold_brain=cfg_sbf['threshold_brain'],
                                    threshold_scalp = cfg_sbf['threshold_scalp'],
                                    sigma_brain=cfg_sbf['sigma_brain'],
                                    sigma_scalp=cfg_sbf['sigma_scalp'])
            
            H = sbf.get_H(G, Adot)            
            Adot = H.copy()

            
        W, D, F, max_eig = calculate_W(Adot, 
                                        lambda_R=lambda_R,
                                        alpha_meas=alpha_meas, 
                                        alpha_spatial=alpha_spatial, 
                                        C_meas_flag=C_meas_flag, 
                                        C_meas=C_meas, 
                                        DIRECT=DIRECT, 
                                        D=D, 
                                        F=F, 
                                        max_eig=max_eig)

        X = _get_image_brain_scalp_indirect(od, W, SB=SB, G=G)

    if 'parcel' in Adot_tmp.coords:
        X = X.assign_coords({"parcel" : ("vertex", Adot_tmp.coords['parcel'].values)})
                            
    if 'is_brain' in Adot_tmp.coords:
        X = X.assign_coords({"is_brain": ("vertex", Adot_tmp.coords['is_brain'].values)}) 
            
    return X, W, D, F, G, max_eig
    

def _get_image_noise_post_direct(A, 
                                W, 
                                lambda_R=1e-6, 
                                alpha_spatial=1e-3, 
                                SB=False, 
                                G=None):
    """
    Compute W and mse_post for a given wavelength using
    spatial regularization (via column scaling) and
    measurement regularization in data space.
    """

    # ---------------------------------------------------------
    # Spatial regularization: R = diag(1 / (B + λ_spatial))
    # ---------------------------------------------------------
    R = _calculate_prior_R(A, alpha_spatial)
    R = R * lambda_R

    # ---------------------------------------------------------
    #  Posterior variance (diagonal only)
    # mse_post(j) = R_j * (1 - (W A^T)_{jj})
    # ---------------------------------------------------------
    s = np.sum(W * A.T, axis=1)   # elementwise multiply row i with column i
    mse_post = R * (1.0 - s)

    if SB:
        mse_post = sbf.go_from_kernel_space_to_image_space_direct(mse_post, G).T
    else:
        split = len(mse_post)//2
        mse_post =  np.reshape( mse_post, (2,split) )

    X_mse_post_xr = xr.DataArray(mse_post, 
                            dims = ['chromo', 'vertex'],
                            coords = {'chromo': ['HbO', 'HbR'] })
    return X_mse_post_xr

def _get_image_noise_post_indirect(A, 
                                W, 
                                lambda_R=1e-6, 
                                alpha_spatial=1e-3, 
                                SB=False, 
                                G=None):
    """
    Compute W and mse_post for a given wavelength using
    spatial regularization (via column scaling) and
    measurement regularization in data space.
    """
    
    lambda_R_indirect = compute_lambda_R_indirect(A, lambda_R, alpha_spatial, A.wavelength)
    mse_lst = []
    # ---------------------------------------------------------
    # 2) Spatial regularization: R = diag(1 / (B + λ_spatial))
    # ---------------------------------------------------------
    for wl in A.wavelength:

        lambda_R_wl = lambda_R_indirect.sel(wavelength=wl).values

        A_wl = A.sel(wavelength=wl).values
        W_wl = W.sel(wavelength=wl).values

        R = _calculate_prior_R(A_wl, alpha_spatial)
        R = R * lambda_R_wl

        # ---------------------------------------------------------
        # Posterior variance (diagonal only)
        # mse_post(j) = R_j * (1 - (W A^T)_{jj})
        # ---------------------------------------------------------
        s = np.sum(W_wl * A_wl.T, axis=1)   # elementwise multiply row i with column i
        mse_post = R * (1.0 - s)

        if SB:
            mse_post = sbf.go_from_kernel_space_to_image_space_indirect(mse_post, G).T

        mse_lst.append(mse_post)

    X_mse_post_xr = xr.DataArray(mse_lst, 
                            dims = ['wavelength', 'vertex'],
                            coords = {'wavelength': A.wavelength })

    return X_mse_post_xr


#%% IMAGE UNCERTAINTY QUANTIFICATION

def get_image_noise_posterior(Adot, 
                            W, 
                            alpha_spatial=1e-3, 
                            lambda_R=1e-6,
                            DIRECT=True, 
                            SB=False, 
                            G=None):
    """
    Compute posterior variance of reconstructed images.
    
    Calculates the diagonal of the posterior covariance matrix:
    Cov(X|y) = R - R * A^T @ (F + λ*C)^(-1) @ A * R
    where R is the spatial prior. Returns only the diagonal (variance at each vertex).
    
    Parameters
    ----------
    Adot : xr.DataArray
        Forward model. For DIRECT: stacked. For INDIRECT: unstacked with wavelength dim.
    W : xr.DataArray
        Inverse matrix from reconstruction.
    alpha_spatial : float, optional
        Spatial regularization weight (default: 1e-3).
    lambda_R : float, optional
        Ridge parameter (default: 1e-6).
    DIRECT : bool, optional
        True for direct method, False for indirect (default: True).
    SB : bool, optional
        Whether spatial basis functions were used (default: False).
    G : dict, optional
        Spatial basis matrices if SB=True.
    
    Returns
    -------
    mse_post : xr.DataArray
        Posterior variance (mean squared error).
        For DIRECT: dimensions (chromo, vertex) with chromophores ['HbO', 'HbR'].
        For INDIRECT: dimensions (chromo, vertex) after conversion from OD space.
        
    Notes
    -----
    Efficient computation using diagonal approximation:
    Var(X_j) = R_j * (1 - sum_i W_ji * A_ij)
    
    For INDIRECT method, computes OD space variance then converts to chromophore
    space using: Var_chromo = E^(-2) @ Var_OD where E is extinction coefficient matrix.
    
    If SB=True, converts from kernel space to image space using G matrices.
    """

    if DIRECT:

        mse_post = _get_image_noise_post_direct(Adot.values, 
                                                W.values,
                                                lambda_R=lambda_R,
                                                alpha_spatial=alpha_spatial, 
                                                SB=SB, 
                                                G=G)

    else: 
        
        E = nirs.get_extinction_coefficients('prahl', Adot.wavelength)
        einv = xrutils.pinv(E)

        mse_post_od = _get_image_noise_post_indirect(Adot, 
                                                    W,
                                                    lambda_R=lambda_R,
                                                    alpha_spatial=alpha_spatial,
                                                    SB=SB, 
                                                    G=G)

        mse_post = einv**2 @ mse_post_od

    return mse_post

#%% PROBE GEOMETRY REGISTRATION

def gen_xform_from_pts(p1, p2):
    """
    Compute affine transformation matrix from two sets of corresponding points.
    
    Finds the (n+1) × (n+1) affine transformation matrix that best maps points
    from p1 to p2 using least-squares: p2 ≈ T @ [p1; 1] where T includes rotation,
    scaling, and translation.
    
    Parameters
    ----------
    p1 : numpy.ndarray
        Source points with shape (n_points, n_dims).
    p2 : numpy.ndarray
        Target points with shape (n_points, n_dims).
    
    Returns
    -------
    t : numpy.ndarray
        Affine transformation matrix with shape (n_dims+1, n_dims+1).
        Last row is [0, 0, ..., 0, 1]. Applies as: p2 = t @ [p1; 1].
        
    Raises
    ------
    ValueError
        If p1 and p2 have different numbers of points or dimensions,
        or if fewer points than dimensions are provided.
        
    Notes
    -----
    Source: https://github.com/bunpc/atlasviewer/blob/71fc98ec/utils/gen_xform_from_pts.m
    
    Uses pseudo-inverse to solve overdetermined system when n_points > n_dims.
    For each dimension i, solves: p2[:,i] = A @ x where A = [p1, ones] and
    x contains rotation/scale and translation parameters.
    """
    p1, p2 = np.array(p1), np.array(p2)
    p = p1.shape[0]
    q = p2.shape[0]
    m = p1.shape[1]
    n = p2.shape[1]
    
    if p != q:
        raise ValueError('number of points for p1 and p2 must be the same')
    
    if m != n:
        raise ValueError('number of dimensions for p1 and p2 must be the same')
    
    if p < n:
        raise ValueError(f'cannot solve transformation with fewer anchor points ({p}) than dimensions ({n}).')
    
    t = np.eye(n + 1)
    a = np.hstack((p1, np.ones((p, 1))))
    
    for ii in range(n):
        x = np.linalg.pinv(a) @ p2[:, ii]
        t[ii, :] = x
        
    return t


def get_probe_aligned(head, geo3d):
    """
    Register probe optodes to head model coordinate system.
    
    Aligns probe source/detector positions to head model using landmark-based
    affine registration. Matches probe landmarks (from 10-10 system) to head
    model landmarks, computes transformation, and snaps optodes to brain surface.
    
    Parameters
    ----------
    head : TwoSurfaceHeadModel
        Head model with brain and scalp surfaces.
    geo3d : xr.DataArray
        Probe geometry containing optode positions (sources, detectors) and
        landmark positions with labels matching 10-10 system (e.g., 'Nz', 'Cz').
    
    Returns
    -------
    probe_snapped_aligned : xr.DataArray
        Probe optode positions in head model coordinates, snapped to brain
        surface. Has dimensions (label, cartesian_axis) with crs='ijk' and
        units in mm.
        
    Notes
    -----
    Registration pipeline:
    1. Load head model fiducials and convert to IJK coordinates
    2. Generate 10-10 landmarks on scalp surface using LandmarksBuilder1010
    3. Find intersection of probe and model landmark labels
    4. Compute affine transformation from matching landmarks
    5. Apply transformation to probe optodes
    6. Snap transformed optodes to nearest brain surface points
    
    Uses gen_xform_from_pts() for affine transformation computation.
    Landmark matching ensures anatomically consistent alignment.
    """

    SEG_DATADIR, mask_files, landmarks_file = datasets.get_icbm152_segmentation()
    masks, t_ijk2ras = io.read_segmentation_masks(SEG_DATADIR, mask_files)

    probe_optodes = geo3d.loc[(geo3d.type==geo.PointType.SOURCE) | (geo3d.type==geo.PointType.DETECTOR)] 
    probe_landmarks = geo3d.loc[geo3d.type==geo.PointType.LANDMARK] 

    # Align fiducials to head coordinate system
    fiducials_ras = io.read_mrk_json(os.path.join(SEG_DATADIR, landmarks_file), crs="aligned")
    t_ijk2ras_inv = np.linalg.pinv(t_ijk2ras)

    t_ijk2ras_inv = t_ijk2ras_inv.pint.quantify('mm')
    t_ijk2ras_inv = t_ijk2ras_inv.rename({'ijk':'tmp', 'aligned':'ijk'})
    t_ijk2ras_inv = t_ijk2ras_inv.rename({'tmp':'aligned'})

    fiducials_ijk = fiducials_ras.points.apply_transform(t_ijk2ras_inv).pint.dequantify().pint.quantify('mm')
    
    # Compute landmarks by EEG's 1010 system rules
    lmbuilder = cgeolm.LandmarksBuilder1010(head.scalp, fiducials_ijk)
    all_landmarks = lmbuilder.build()

    model_ref_pos = np.array(all_landmarks)  
    model_ref_labels = [lab.item() for lab in all_landmarks.label] 

    probe_landmark_pos = list(np.array(probe_landmarks.values))
    probe_landmark_labels = list(np.array(probe_landmarks.label))

    # Construct transform from intersection
    intersection = list(set(probe_landmark_labels) & set(model_ref_labels)) 
    model_ref_pos = [model_ref_pos[model_ref_labels.index(intsct)] for intsct in intersection]
    probe_ref_pos = [probe_landmark_pos[probe_landmark_labels.index(intsct)] for intsct in intersection]

    T = gen_xform_from_pts(probe_ref_pos, model_ref_pos)

    probe_aligned = probe_optodes.points.apply_transform(T)
    probe_aligned = probe_aligned.points.set_crs('ijk')
    probe_aligned = probe_aligned.pint.dequantify().pint.quantify('mm')

    # Snap to surface
    probe_snapped_aligned = head.brain.snap(probe_aligned)

    return probe_snapped_aligned


#%% DEPRECATED CODE (kept for reference)
# The following function was an earlier implementation of image noise estimation
# that computed covariance by propagating measurement noise through W matrix.
# Current implementation uses posterior variance formulation for efficiency.

    # def get_image_noise(C_meas, X, W, SB=False, DIRECT=True, G=None):

    # TIME = False
    # if 'time' in C_meas.dims:
    #     t_dim = 'time'
    #     TIME = True
    # elif 'reltime' in C_meas.dims:
    #     t_dim = 'reltime'
    #     TIME = True

    # if DIRECT:
    #     if TIME:
    #         C_tmp_lst = []
    #         for t in C_meas[t_dim]:
    #                 C_tmp = C_meas.sel({t_dim:t})
    #                 cov_img_tmp = W * np.sqrt(C_tmp.values) # get diag of image covariance
    #                 cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
    #                 C_tmp_lst.append(cov_img_diag)

    #         cov_img_diag = np.vstack(C_tmp_lst)
    #     else:
    #         cov_img_tmp = W *np.sqrt(C_meas.values) # W is pseudo inverse  --- diagonal (faster than W C W.T)
    #         cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
        
    #     if SB:
    #         cov_img_diag = sbf.go_from_kernel_space_to_image_space_direct(cov_img_diag, G)
    #     else:
    #         if TIME:
    #             split = cov_img_diag.shape[1]//2
    #             HbO = cov_img_diag[:, :split]   # shape: (time, vertices)
    #             HbR = cov_img_diag[:, split:]   # shape: (time, vertices)

    #             # Stack into vertex x 2 x time
    #             cov_img_diag = np.stack([HbO.T, HbR.T], axis=1)  # (vertex, 2, time)
    #             cov_img_diag = cov_img_diag.transpose(0,2,1)
    #         else:
    #             split = len(cov_img_diag)//2
    #             cov_img_diag =  np.reshape( cov_img_diag, (2,split) ).T 
        

    # else:
    #     cov_img_lst = []
    #     E = nirs.get_extinction_coefficients('prahl', W.wavelength)
    #     einv = xrutils.pinv(E)

    #     for wavelength in W.wavelength:
    #         W_wl = W.sel(wavelength=wavelength)
    #         C_wl = C_meas.sel(wavelength=wavelength)

    #         if TIME:
    #             C_tmp_lst = []
    #             for t in C_wl[t_dim]:
    #                 C_tmp = C_wl.sel({t_dim:t})
    #                 cov_img_tmp = W_wl * np.sqrt(C_tmp.values) # get diag of image covariance
    #                 cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
    #                 C_tmp_lst.append(cov_img_diag)

    #             cov_img_diag = np.vstack(C_tmp_lst)

    #         else:
    #             cov_img_tmp = W_wl * np.sqrt(C_wl.values) # get diag of image covariance
    #             cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
            
    #         if SB:
    #             cov_img_diag = sbf.go_from_kernel_space_to_image_space_indirect(cov_img_diag, G)
            
    #         cov_img_lst.append(cov_img_diag)
            
    #     if TIME:
    #         cov_img_diag =  np.stack(cov_img_lst, axis=2) 
    #         cov_img_diag = np.transpose(cov_img_diag, [2,1,0])
    #         cov_img_diag = np.einsum('ij,jab->iab', einv.values**2, cov_img_diag)

    #     else:
    #         cov_img_diag =  np.vstack(cov_img_lst) 
    #         cov_img_diag = einv.values**2 @ cov_img_diag

    # noise = X.copy()
    # noise.values = cov_img_diag

    # return noise