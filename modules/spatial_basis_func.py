#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spatial_basis_func.py

Spatial basis function utilities for fNIRS image reconstruction. This module
provides tools for creating spatially-informed basis functions on brain and
scalp surfaces to improve image reconstruction quality and reduce dimensionality.

Key Functionality:
- Sensitivity masking: Identify vertices with sufficient forward model sensitivity
- Mesh downsampling: Generate sparse seed points for basis function centers
- Kernel matrices: Create Gaussian spatial basis functions centered at seeds
- Forward model transformation: Apply spatial basis to forward matrices
- Space conversions: Transform between kernel and image space representations

Spatial basis functions reduce reconstruction ill-posedness by:
1. Constraining solutions to smooth spatial patterns
2. Reducing the number of unknowns from full vertex count to kernel count
3. Incorporating anatomical priors through surface-based Gaussian kernels

Initial Contributors:
- Yuanyuan Gao
- Laura Carlton | lcarlton@bu.edu | 2024
"""

import numpy as np
import xarray as xr
from scipy.spatial import KDTree
from tqdm import tqdm

import cedalion
import cedalion.imagereco.forward_model as cfm


#%% GETTING THE SPATIAL BASIS 

def get_sensitivity_mask(sensitivity: xr.DataArray, threshold: float = -2, wavelength_idx: int = 0):
    """
    Generate a binary mask indicating vertices with sufficient forward model sensitivity.
    
    Creates a mask based on the log10 of summed sensitivity across channels. Vertices
    with log-sensitivity above the threshold are marked for inclusion in reconstruction.
    This helps exclude vertices with poor measurement coupling.
    
    Parameters
    ----------
    sensitivity : xr.DataArray
        Forward model sensitivity matrix with dimensions (channel, vertex, wavelength).
    threshold : float, optional
        Log10 threshold for sensitivity (default: -2, corresponding to 1% sensitivity).
    wavelength_idx : int, optional
        Wavelength index to use for mask computation if multiple wavelengths present
        (default: 0).
    
    Returns
    -------
    mask : xr.DataArray
        Boolean mask with dimension (vertex,) containing True for vertices with
        log10(sum_sensitivity) > threshold.
        
    Notes
    -----
    The wavelength coordinate is dropped from the returned mask.
    """
   
    intensity = np.log10(sensitivity[:,:,wavelength_idx].sum('channel'))
    mask = intensity > threshold
    mask = mask.drop_vars('wavelength')
    
    return mask


def downsample_mesh(mesh: xr.DataArray, 
                    mask: xr.DataArray,
                    threshold: cedalion.Quantity = 5 * cedalion.units.mm):
    """
    Downsample mesh surface to generate sparse seed points for spatial basis functions.
    
    Uses a greedy distance-based algorithm: iteratively selects vertices from the masked
    mesh ensuring each new vertex is at least 'threshold' distance from all previously
    selected vertices. This creates a spatially uniform distribution of basis function
    centers.
    
    Parameters
    ----------
    mesh : xr.DataArray
        Surface mesh vertices with dimensions (label, cartesian_axis) containing
        3D coordinates in mm.
    mask : xr.DataArray
        Boolean mask with dimension (label,) indicating which vertices have sufficient
        sensitivity for inclusion.
    threshold : cedalion.Quantity, optional
        Minimum distance between selected vertices in the downsampled mesh
        (default: 5 mm).
    
    Returns
    -------
    mesh_downsampled : xr.DataArray
        Downsampled mesh with dimensions (vertex, cartesian_axis) containing 3D
        coordinates of selected seed vertices.
        
    Notes
    -----
    Uses KDTree for efficient nearest-neighbor queries. The algorithm processes
    vertices sequentially and rebuilds the KDTree after each addition.
    
    Initial Contributors:
        - Yuanyuan Gao 
        - Laura Carlton | lcarlton@bu.edu | 2024
    """
    
    mesh_units = mesh.pint.units
    threshold = threshold.to(mesh_units)
    
    mesh = mesh.rename({'label':'vertex'}).pint.dequantify()
    mesh_masked = mesh[mask,:]
    mesh_new = []

    for vv in tqdm(mesh_masked):
        if len(mesh_new) == 0: 
            mesh_new.append(vv)
            tree = KDTree(mesh_new)
            continue
        
        distance, _ = tree.query(vv, distance_upper_bound=threshold.magnitude)
        
        if distance == float('inf'):
            mesh_new.append(vv)
            tree = KDTree(mesh_new)

    
    mesh_new_xr = xr.DataArray(mesh_new,
                               dims = mesh.dims,
                               coords = {'vertex':np.arange(len(mesh_new))},
                               attrs = {'units': mesh_units }
        )
    
    mesh_new_xr = mesh_new_xr.pint.quantify()
    
    return mesh_new_xr




def get_kernel_matrix(mesh_downsampled: xr.DataArray, 
                      mesh: xr.DataArray, 
                      sigma: cedalion.Quantity = 5 * cedalion.units.mm):
    """
    Construct matrix of Gaussian spatial basis functions on surface mesh.
    
    Creates a kernel matrix G where each column represents a 3D isotropic Gaussian
    basis function centered at a downsampled vertex. The value at each full-mesh
    vertex is computed using the multivariate Gaussian probability density function.
    This encodes spatial smoothness: nearby vertices receive similar weights.
    
    Parameters
    ----------
    mesh_downsampled : xr.DataArray
        Sparse mesh vertices (basis function centers) with dimensions
        (vertex, cartesian_axis) containing 3D coordinates.
    mesh : xr.DataArray
        Full-resolution mesh vertices with dimensions (label, cartesian_axis)
        containing 3D coordinates. Must have same units as mesh_downsampled.
    sigma : cedalion.Quantity, optional
        Standard deviation of Gaussian kernels in mm (default: 5 mm). Controls
        the spatial smoothness scale.
    
    Returns
    -------
    kernel_matrix : xr.DataArray
        Kernel matrix with dimensions (kernel, vertex) or (vertex, kernel)
        where G[i,j] is the weight of kernel i at vertex j. Rows sum to
        approximate unity (Gaussian integral).
        
    Notes
    -----
    Uses isotropic covariance: Σ = σ²I where I is 3×3 identity matrix.
    Computation is vectorized using einsum for efficiency.
    
    Initial Contributors:
        - Yuanyuan Gao 
        - Laura Carlton | lcarlton@bu.edu | 2024
    """
    assert mesh.pint.units == mesh_downsampled.pint.units
    
    mesh_units = mesh.pint.units
    sigma = sigma.to(mesh_units)
    
    cov_matrix = (sigma.magnitude ** 2) * np.eye(3)
    inv_cov = np.linalg.inv(cov_matrix)
    det_cov = np.linalg.det(cov_matrix)
    denominator = np.sqrt((2 * np.pi) ** 3 * det_cov)

    mesh_downsampled = mesh_downsampled.pint.dequantify().values
    mesh = mesh.pint.dequantify().values
    
    diffs = mesh_downsampled[:, None, :] - mesh[None, :, :]
    
    # Compute Mahalanobis distance: (x-μ)ᵀ Σ⁻¹ (x-μ) for all pairs
    exponents = -0.5 * np.einsum('ijk,kl,ijl->ij', diffs, inv_cov, diffs)
    
    kernel_matrix = np.exp(exponents) / denominator
    n_vertex = mesh.shape[0]
    
    dimensions = kernel_matrix.shape
    
    if dimensions[0] != n_vertex:
        dims = ["kernel", "vertex"]
        n_kernel = dimensions[0]
    else:
        dims = ["vertex", "kernel"]
        n_kernel = dimensions[1]
    
    kernel_matrix_xr = xr.DataArray(kernel_matrix, 
                                    dims = dims,
                                    coords = {'vertex': np.arange(n_vertex),
                                              'kernel': np.arange(n_kernel)}
                                    )
    
    return kernel_matrix_xr



def get_G_matrix(head: cfm.TwoSurfaceHeadModel, 
                 M: xr.DataArray,
                 threshold_brain: cedalion.Quantity = 1 * cedalion.units.mm, 
                 threshold_scalp: cedalion.Quantity = 5 * cedalion.units.mm, 
                 sigma_brain: cedalion.Quantity = 1 * cedalion.units.mm, 
                 sigma_scalp: cedalion.Quantity = 5 * cedalion.units.mm
                 ):
    """
    Construct spatial basis matrices for both brain and scalp surfaces.
    
    This is the main wrapper function that creates separate spatial basis function
    representations for brain and scalp tissue layers. Each layer is downsampled
    independently and has its own Gaussian kernel parameters, allowing different
    spatial smoothness scales for each tissue type.
    
    Parameters
    ----------
    head : cfm.TwoSurfaceHeadModel
        Two-layer head model containing brain and scalp surface meshes.
    M : xr.DataArray
        Boolean sensitivity mask with dimensions (vertex,) and coordinate 'is_brain'
        indicating which vertices have sufficient forward model sensitivity.
    threshold_brain : cedalion.Quantity, optional
        Minimum distance between brain seed vertices (default: 1 mm).
    threshold_scalp : cedalion.Quantity, optional
        Minimum distance between scalp seed vertices (default: 5 mm).
    sigma_brain : cedalion.Quantity, optional
        Standard deviation of brain Gaussian kernels (default: 1 mm).
        Smaller values give better spatial resolution.
    sigma_scalp : cedalion.Quantity, optional
        Standard deviation of scalp Gaussian kernels (default: 5 mm).
        Typically larger than brain to reflect diffuse scalp signals.
    
    Returns
    -------
    G : dict
        Dictionary containing:
        - 'G_brain': xr.DataArray with dimensions (kernel, vertex) for brain
        - 'G_scalp': xr.DataArray with dimensions (kernel, vertex) for scalp
        
    Notes
    -----
    Brain and scalp are processed independently based on the 'is_brain' coordinate
    in the mask M. This allows asymmetric treatment reflecting different tissue
    properties.
    
    Initial Contributors:
        - Yuanyuan Gao 
        - Laura Carlton | lcarlton@bu.edu | 2024
    """
    
    brain_downsampled = downsample_mesh(head.brain.vertices, M[M.is_brain], threshold_brain)
    scalp_downsampled = downsample_mesh(head.scalp.vertices, M[~M.is_brain], threshold_scalp)
    
    G_brain = get_kernel_matrix(brain_downsampled, head.brain.vertices, sigma_brain)
    G_scalp = get_kernel_matrix(scalp_downsampled, head.scalp.vertices, sigma_scalp)
    

    G = {'G_brain': G_brain, 
         'G_scalp': G_scalp
         }
    
    return G

#%% TRANSFORMING FORWARD MODEL: H = A @ G

def get_H(G, A):
    """
    Transform forward model to spatial basis representation (single-wavelength per matrix).
    
    Computes H = A @ G where:
    - A is the forward model mapping vertex activations to measurements
    - G is the spatial basis matrix mapping kernel weights to vertex values
    - H maps kernel weights directly to measurements
    
    This reduces the reconstruction problem from n_vertices unknowns to n_kernels
    unknowns. Processes brain and scalp separately for each wavelength.
    
    Parameters
    ----------
    G : dict
        Dictionary containing:
        - 'G_brain': xr.DataArray with dimensions (kernel, vertex)
        - 'G_scalp': xr.DataArray with dimensions (kernel, vertex)
    A : xr.DataArray
        Forward model sensitivity matrix with dimensions (channel, vertex, wavelength)
        and coordinate 'is_brain' indicating brain vs scalp vertices.
    
    Returns
    -------
    H : xr.DataArray
        Transformed forward model with dimensions (channel, kernel, wavelength).
        First n_brain_kernels correspond to brain, remaining to scalp.
        Includes 'is_brain' coordinate indicating kernel type.
        
    Notes
    -----
    Maintains separation between wavelengths - does not stack them.
    Brain kernels are listed first, followed by scalp kernels.
    """
    n_channel = A.shape[0]
    nV_brain = A.is_brain.sum().values 

    nkernels_brain = G['G_brain'].kernel.shape[0]
    nkernels_scalp = G['G_scalp'].kernel.shape[0]
    n_kernels = nkernels_brain + nkernels_scalp

    is_brain = np.zeros(n_kernels, dtype=bool)
    is_brain[:nkernels_brain] = True

    H = np.zeros( (n_channel, n_kernels, 2))

    for w_idx, wl in enumerate(A.wavelength):
        A_wl = A.sel(wavelength=wl)
        A_wl_brain = A_wl[:,:nV_brain]
        A_wl_scalp = A_wl[:,nV_brain:]

        H[:,:nkernels_brain, w_idx] = A_wl_brain.values @ G['G_brain'].values.T
        
        H[:, nkernels_brain:, w_idx] = A_wl_scalp.values @ G['G_scalp'].values.T

    H = xr.DataArray(H, dims=("channel", "kernel", "wavelength"))
    H = H.assign_coords({'channel': A.channel,
                         'wavelength': A.wavelength,
                         'is_brain': ('kernel', is_brain)})
    
    return H

    
def get_H_stacked(G, A):
    """
    Transform forward model to spatial basis for stacked dual-wavelength system.
    
    Computes H = A @ G_stacked for direct chromophore reconstruction where the
    forward model A has been pre-multiplied by the extinction coefficient matrix
    and stacked as [A_HbO; A_HbR]. The output maps from chromophore kernel weights
    to measurements.
    
    Parameters
    ----------
    G : dict
        Dictionary containing:
        - 'G_brain': xr.DataArray with dimensions (kernel, vertex)
        - 'G_scalp': xr.DataArray with dimensions (kernel, vertex)
    A : xr.DataArray
        Stacked forward model with dimensions (channel, vertex) where vertex
        dimension contains [HbO_brain, HbO_scalp, HbR_brain, HbR_scalp] in order.
        The 'is_brain' coordinate has same stacking pattern.
    
    Returns
    -------
    H : xr.DataArray
        Transformed forward model with dimensions (channel, kernel) where kernel
        dimension contains [HbO_brain_kernels, HbO_scalp_kernels, HbR_brain_kernels,
        HbR_scalp_kernels] matching the input stacking pattern.
        
    Notes
    -----
    This is used for direct chromophore reconstruction where the inverse problem
    is solved directly in HbO/HbR space rather than wavelength space.
    """
    n_channel = A.shape[0]
    nV_brain = A.is_brain.sum().values //2
    nV_scalp = (~A.is_brain).sum().values //2

    nkernels_brain = G['G_brain'].kernel.shape[0]
    nkernels_scalp = G['G_scalp'].kernel.shape[0]

    n_kernels = nkernels_brain + nkernels_scalp

    H = np.zeros( (n_channel, 2 * n_kernels))

    A_hbo_brain = A[:, :nV_brain]
    A_hbr_brain = A[:, nV_brain+nV_scalp:2*nV_brain+nV_scalp]
    
    A_hbo_scalp = A[:, nV_brain:nV_scalp+nV_brain]
    A_hbr_scalp = A[:, 2*nV_brain+nV_scalp:]
    
    H[:,:nkernels_brain] = A_hbo_brain.values @ G['G_brain'].values.T
    H[:, nkernels_brain+nkernels_scalp:2*nkernels_brain+nkernels_scalp] = A_hbr_brain.values @ G['G_brain'].values.T
    
    H[:, nkernels_brain:nkernels_brain+nkernels_scalp] = A_hbo_scalp.values @ G['G_scalp'].values.T
    H[:, 2*nkernels_brain+nkernels_scalp:] = A_hbr_scalp.values @ G['G_scalp'].values.T

    H = xr.DataArray(H, dims=("channel", "kernel"))
    
    return H


def go_from_kernel_space_to_image_space_direct(X, G):
    """
    Transform direct chromophore reconstruction from kernel space to image space.
    
    Applies spatial basis matrices to convert kernel weights back to vertex-wise
    chromophore concentrations. The input is in stacked form [HbO_kernels; HbR_kernels]
    and output is [HbO_vertices; HbR_vertices].
    
    Parameters
    ----------
    X : numpy.ndarray
        Reconstructed kernel weights with shape (2*n_kernels,) or (2*n_kernels, time).
        First half are HbO kernel weights (brain then scalp), second half are HbR
        kernel weights (brain then scalp).
    G : dict
        Dictionary containing:
        - 'G_brain': xr.DataArray with dimensions (kernel, vertex)
        - 'G_scalp': xr.DataArray with dimensions (kernel, vertex)
    
    Returns
    -------
    X_image : numpy.ndarray
        Reconstructed image in vertex space with shape (n_vertices, 2) for 1D input
        or (n_vertices, 2, time) for 2D input. Last dimension is [HbO, HbR].
        Vertices are ordered [brain, scalp] for each chromophore.
        
    Notes
    -----
    Transformation: x_vertex = G.T @ x_kernel
    Processes HbO and HbR separately, each split into brain and scalp components.
    """

    split = X.shape[0]//2
    nkernels_brain = G['G_brain'].kernel.shape[0]
    if len(X.shape) > 1:
        X_hbo = X[:split, :]
        X_hbr = X[split:, :]
        sb_X_brain_hbo = X_hbo[:nkernels_brain, :]
        sb_X_brain_hbr = X_hbr[:nkernels_brain, :]
        sb_X_scalp_hbo = X_hbo[nkernels_brain:, :]
        sb_X_scalp_hbr = X_hbr[nkernels_brain:, :]
    else:
        X_hbo = X[:split]
        X_hbr = X[split:]
        sb_X_brain_hbo = X_hbo[:nkernels_brain]
        sb_X_brain_hbr = X_hbr[:nkernels_brain]
        sb_X_scalp_hbo = X_hbo[nkernels_brain:]
        sb_X_scalp_hbr = X_hbr[nkernels_brain:]
    
    # Project back to surface space 
    X_hbo_brain = G['G_brain'].values.T @ sb_X_brain_hbo
    X_hbo_scalp = G['G_scalp'].values.T @ sb_X_scalp_hbo
    X_hbr_brain = G['G_brain'].values.T @ sb_X_brain_hbr
    X_hbr_scalp = G['G_scalp'].values.T @ sb_X_scalp_hbr
    
    if len(X.shape) == 1:
        X = np.stack([np.concatenate([X_hbo_brain, X_hbo_scalp]), 
                      np.concatenate([X_hbr_brain, X_hbr_scalp])], axis=1)
    else:
        X = np.stack([np.vstack([X_hbo_brain, X_hbo_scalp]), 
                      np.vstack([X_hbr_brain, X_hbr_scalp])], axis=2)

    return X

def go_from_kernel_space_to_image_space_indirect(X, G):
    """
    Transform indirect (wavelength-space) reconstruction from kernel space to image space.
    
    Applies spatial basis matrices to convert kernel weights back to vertex-wise
    optical density changes. Used for single-wavelength or wavelength-by-wavelength
    reconstruction before conversion to chromophore concentrations.
    
    Parameters
    ----------
    X : numpy.ndarray
        Reconstructed kernel weights with shape (n_kernels,) or (n_kernels, time).
        First n_brain_kernels are brain, remaining are scalp.
    G : dict
        Dictionary containing:
        - 'G_brain': xr.DataArray with dimensions (kernel, vertex)
        - 'G_scalp': xr.DataArray with dimensions (kernel, vertex)
    
    Returns
    -------
    X_image : numpy.ndarray
        Reconstructed image in vertex space with shape (n_vertices,) for 1D input
        or (n_vertices, time) for 2D input. Vertices are ordered [brain, scalp].
        
    Notes
    -----
    Transformation: x_vertex = G.T @ x_kernel
    Simpler than direct method since only one wavelength/component at a time.
    """
    
    nkernels_brain = G['G_brain'].kernel.shape[0]
    if len(X.shape) < 2:
        sb_X_brain = X[:nkernels_brain]
        sb_X_scalp = X[nkernels_brain:]
    else:
        sb_X_brain = X[:nkernels_brain, :]
        sb_X_scalp = X[nkernels_brain:, :]
    
    # Project back to surface space
    X_brain = G['G_brain'].values.T @ sb_X_brain
    X_scalp = G['G_scalp'].values.T @ sb_X_scalp
    
    X = np.concatenate([X_brain, X_scalp])
    
    return X





