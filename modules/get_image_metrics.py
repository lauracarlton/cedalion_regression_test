#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_image_metrics.py

Image quality metrics module for fNIRS image reconstruction evaluation. This
module provides functions to compute various quality metrics for reconstructed
fNIRS images including spatial resolution (FWHM), contrast-to-noise ratio (CNR),
localization error, crosstalk between tissue layers and chromophores, and 
geometric measures.

Key Metrics:
- FWHM: Full width at half maximum - spatial resolution measure
- CNR: Contrast-to-noise ratio - signal quality measure  
- Localization error: Distance between true and reconstructed activation centers
- Crosstalk: Interference between tissue layers or chromophores
- Percent reconstructed: Signal distribution between brain and scalp

Author: Laura Carlton
Created: October 30, 2024
"""

import sys

import numpy as np
import xarray as xr
from scipy.spatial.distance import cdist

from cedalion import nirs, units, xrutils

sys.path.append('/projectnb/nphfnirs/s/users/lcarlton/ANALYSIS_CODE/imaging_paper_figure_code/modules/')
import spatial_basis_func as sbf


#%%

def get_ROI(image, threshold = 0.5):
    """
    Extract region of interest (ROI) from an image based on amplitude threshold.
    
    Identifies vertices where the image amplitude exceeds a specified fraction
    of the maximum amplitude.
    
    Parameters
    ----------
    image : numpy.ndarray or xarray.DataArray
        Reconstructed image values at each vertex.
    threshold : float, optional
        Fraction of maximum amplitude for ROI inclusion (default: 0.5).
        
    Returns
    -------
    ROI : numpy.ndarray
        Array of vertex indices where amplitude > threshold * max_amplitude.
    """
    max_vertex_amp = image.max()
    ROI = np.where(image > max_vertex_amp*threshold)[0]
    
    return ROI

def get_ROI_contig(image, head, threshold = 0.5):
    """
    Extract spatially contiguous region of interest containing the peak vertex.
    
    First identifies ROI vertices above threshold, then extracts only the 
    contiguous region connected to the maximum amplitude vertex.
    
    Parameters
    ----------
    image : numpy.ndarray or xarray.DataArray
        Reconstructed image values at each vertex.
    head : Head model object
        Contains mesh topology (vertices, faces) for determining connectivity.
    threshold : float, optional
        Fraction of maximum amplitude for initial ROI (default: 0.5).
        
    Returns
    -------
    ROI_contig : list
        Vertex indices forming contiguous region around peak.
    """
    ROI = get_ROI(image, threshold=threshold)
    ROI_contig = get_contiguous_blob(image.argmax(), head, ROI)
    
    return ROI_contig
    
    
def get_image_centroid(image, ROI, head):
    """
    Compute amplitude-weighted centroid of image within ROI.
    
    Calculates the center of mass of the activation using vertex positions
    weighted by their image amplitudes.
    
    Parameters
    ----------
    image : numpy.ndarray or xarray.DataArray
        Reconstructed image values at each vertex.
    ROI : numpy.ndarray or list
        Vertex indices defining the region of interest.
    head : Head model object
        Contains mesh vertices for position information.
        
    Returns
    -------
    centroid : numpy.ndarray
        3D coordinates [1, 3] of the weighted centroid.
    """
    if len(ROI) == 1:
        weights = image[ROI]
    else:
        weights = image[ROI].squeeze()
        
    centroid = np.average(head.brain.mesh.vertices[ROI,:], axis=0, weights=weights)
    centroid = np.reshape(centroid, [1, len(centroid)])
    
    return centroid
        
    
def get_FWHM(image, head, ROI_threshold = 0.5, version='weighted_mean'):
    """
    Compute Full Width at Half Maximum (FWHM) as a spatial resolution metric.
    
    FWHM quantifies the spatial extent of the reconstructed activation blob.
    Three computation methods are available:
    - 'mean': Average distance from vertices to centroid
    - 'max': Maximum pairwise distance in contiguous ROI  
    - 'weighted_mean': Amplitude-weighted distance (Gaussian approximation)
    
    Parameters
    ----------
    image : numpy.ndarray or xarray.DataArray
        Reconstructed image values at each vertex.
    head : Head model object
        Contains mesh structure for geometric calculations.
    ROI_threshold : float, optional
        Fraction of maximum amplitude for ROI definition (default: 0.5).
    version : str, optional
        Computation method: 'mean', 'max', or 'weighted_mean' (default: 'weighted_mean').
        
    Returns
    -------
    FWHM : float
        Full width at half maximum in mm. For 'weighted_mean', applies Gaussian
        conversion factor: FWHM = 2 * sqrt(2*ln(2)) * weighted_mean_distance.
    """

    # get contiuguous region containing the max vertex
    # ROI_contig = get_contiguous_blob(max_vertex, head, ROI)

    # get mean distance between centroid of contig blob and all vertices in blob
    
    if version == 'mean':
        ROI = get_ROI(image, threshold=ROI_threshold)  

        ## find the centroid ##    
        centroid = get_image_centroid(image, ROI, head)
        
        ## find average distance from all points in ROI to the centroid ##
        distances = np.linalg.norm(centroid - head.brain.mesh.vertices[ROI,:], axis = 1)
        FWHM = np.mean(distances)
    
    if version == 'max':
        
        ROI_contig = get_ROI_contig(image, head, threshold=ROI_threshold)
        
        ## get all the pairwise distances
        distances = cdist(head.brain.mesh.vertices[ROI_contig,:],head.brain.mesh.vertices[ROI_contig,:])
        
        ## take the max distance
        FWHM = np.max(distances)
        
    if version == 'weighted_mean':
        
        ROI = get_ROI(image, threshold=ROI_threshold)
        centroid = get_image_centroid(image, ROI, head)
        img_sum = np.sum(image[ROI])
        
        # get the weighted average distance between all vertices in ROI and the centroid 
        distances = np.linalg.norm(centroid - head.brain.mesh.vertices[ROI,:], axis=1)
        FWHM = np.sum(distances * image[ROI])/img_sum
        FWHM = 2* np.sqrt(2*np.log(2)) * FWHM

        
    return FWHM



def get_crosstalk(image_brain, image_scalp, ROI_threshold=0.5):
    """
    Compute crosstalk between brain and scalp tissue layers.
    
    Quantifies the degree to which signal intended for one tissue layer
    appears in another layer. Two measures are computed:
    - Mean-based: ratio of mean amplitudes in each ROI
    - RMS-based: ratio of root-mean-square amplitudes
    
    Parameters
    ----------
    image_brain : numpy.ndarray or xarray.DataArray
        Reconstructed image values for brain vertices.
    image_scalp : numpy.ndarray or xarray.DataArray
        Reconstructed image values for scalp vertices.
    ROI_threshold : float, optional
        Fraction of maximum for ROI definition (default: 0.5).
        
    Returns
    -------
    crosstalk_max : float
        Ratio of mean scalp amplitude to mean brain amplitude in respective ROIs.
    crosstalk_rms : float
        Ratio of RMS scalp amplitude to RMS brain amplitude in respective ROIs.
    """
    
    ROI_scalp = get_ROI(image_scalp, threshold=ROI_threshold)
    ROI_brain = get_ROI(image_brain, threshold=ROI_threshold)
    
    # get crosstalk using the max of each image
    contrast_brain = np.mean(image_brain[ROI_brain])
    contrast_scalp = np.mean(image_scalp[ROI_scalp])
    
    crosstalk_max = contrast_scalp/contrast_brain
    
    # get the crosstalk using the RMS of each image 
    contrast_brain = np.sqrt(np.mean(image_brain[ROI_brain]**2))
    contrast_scalp = np.sqrt(np.mean(image_scalp[ROI_scalp]**2))
    
    crosstalk_rms = contrast_scalp/contrast_brain
    
    return crosstalk_max, crosstalk_rms


def get_CNR_hbo(image, y, W, ROI_threshold=0.5, n_noise_instances=25, perc_noise=0.01, SB=False, G=None, DIRECT_flag=True):
    """
    Compute Contrast-to-Noise Ratio (CNR) for HbO chromophore in dual-wavelength reconstruction.
    
    Uses Monte Carlo simulation to estimate noise by adding random measurement
    noise and computing the resulting image variability. Supports both direct
    (chromophore space) and indirect (wavelength space) reconstruction methods,
    with optional spatial basis functions.
    
    Parameters
    ----------
    image : xarray.DataArray
        Noise-free reconstructed image with 'chromo' dimension.
    y : numpy.ndarray
        Measurement vector (may be stacked wavelengths for indirect method).
    W : xarray.DataArray
        Inverse matrix (reconstruction operator).
    ROI_threshold : float, optional
        Fraction of max for ROI definition (default: 0.5).
    n_noise_instances : int, optional
        Number of Monte Carlo noise realizations (default: 25).
    perc_noise : float, optional
        Standard deviation of Gaussian measurement noise (default: 0.01).
    SB : bool, optional
        Whether spatial basis functions are used (default: False).
    G : dict, optional
        Dictionary with 'G_brain' and 'G_scalp' kernel matrices if SB=True.
    DIRECT_flag : bool, optional
        True for direct chromophore reconstruction, False for indirect (default: True).
        
    Returns
    -------
    CNR : float
        Contrast-to-noise ratio (mean contrast / std of noise realizations).
    contrast_brain : float
        Mean HbO amplitude in brain ROI.
    noise_mean : float
        Mean of noise realizations.
    max_values : list
        Maximum values from each noise realization.
    noise_instances : list
        Mean ROI values from each noise realization.
    image_max : float
        Maximum amplitude in noise-free image.
    """
    
    image_hbo = image.sel(chromo='HbO')
    ROI = get_ROI(image_hbo, threshold=ROI_threshold)
    max_vertex_idx = image_hbo[ROI].argmax()
    contrast_brain = image_hbo[ROI].mean()
    split = len(y)//2
    n_brain = len(image_hbo)
    # run many instances of noise
    noise_instances = []
    max_values = []
    
    for kk in range(n_noise_instances):
        meas_noise = np.random.normal(0, perc_noise, len(y))  #np.random.normal(0, perc_noise, len(y)) #* perc_noise #  gaussian noise 
        y_noise = y + meas_noise
    
        if DIRECT_flag:
            
            image_noise = W.values @ y_noise
            
            if SB:
                image_noise = sbf.go_from_kernel_space_to_image_space_direct(image_noise, G)
                image_hbo = image_noise[:,0]
                image_noise_hbo = image_hbo[:n_brain]
            else:
                image_noise_brain = image_noise[W.is_brain.values]
                img_split = len(image_noise_brain)//2
                image_noise_hbo = image_noise_brain[:img_split]
            
                
        else:
            W_indirect_wl0 = W.isel(wavelength=0)
            W_indirect_wl1 = W.isel(wavelength=1)
            
            y_wl0 = y_noise[:split]
            X_wl0 = W_indirect_wl0.values @ y_wl0
          
            y_wl1 = y_noise[split:]
            X_wl1 = W_indirect_wl1.values @ y_wl1
                    
            if SB:
                X_wl0 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl0, G)
                X_wl1 = sbf.go_from_kernel_space_to_image_space_indirect(X_wl1, G)
                
                
            X_od = np.vstack([X_wl0, X_wl1]).T
            E = nirs.get_extinction_coefficients('prahl', W.wavelength)
            einv = xrutils.pinv(E) #FIXME check unit
            X_od = xr.DataArray(X_od, dims=('vertex', 'wavelength'), coords={'wavelength':W.wavelength} )
            image_noise = xr.dot(einv, X_od/units.mm, dims=["wavelength"])
            image_noise_hbo = image_noise.sel(chromo='HbO')[:n_brain].values
        
        noise_instances.append(np.mean(image_noise_hbo[ROI]))
        max_values.append(image_noise_hbo[max_vertex_idx])
    
    noise_std = np.std(np.asarray(noise_instances))
    noise_mean = np.mean(np.asarray(noise_instances))
    
    CNR = contrast_brain/noise_std
    
    return CNR, contrast_brain, noise_mean, max_values, noise_instances, image.max()


def get_CNR(image, y, W, n_brain, head, ROI_threshold=0.5, n_noise_instances=25, perc_noise=0.01, SB=False, G=None):
    """
    Compute Contrast-to-Noise Ratio (CNR) for single-wavelength reconstruction.
    
    Uses Monte Carlo simulation with multiple noise realizations to estimate
    the variability in reconstructed amplitude. Supports optional spatial
    basis function reconstruction.
    
    Parameters
    ----------
    image : numpy.ndarray or xarray.DataArray
        Noise-free reconstructed image.
    y : numpy.ndarray
        Measurement vector.
    W : numpy.ndarray or xarray.DataArray
        Inverse matrix (reconstruction operator).
    n_brain : int
        Number of brain vertices (for separating brain from scalp).
    head : Head model object
        Contains mesh structure (not actively used in function).
    ROI_threshold : float, optional
        Fraction of max for ROI definition (default: 0.5).
    n_noise_instances : int, optional
        Number of Monte Carlo realizations (default: 25).
    perc_noise : float, optional
        Standard deviation of Gaussian measurement noise (default: 0.01).
    SB : bool, optional
        Whether spatial basis functions are used (default: False).
    G : dict, optional
        Dictionary with 'G_brain' kernel matrix if SB=True.
        
    Returns
    -------
    CNR : float
        Contrast-to-noise ratio (contrast / std of noise realizations).
    contrast_brain : float
        Mean amplitude in brain ROI.
    noise_mean : float
        Mean value across noise realizations.
    max_values : list
        Peak vertex values from each noise realization.
    noise_instances : list
        Mean ROI values from each noise realization.
    image_max : float
        Maximum amplitude in noise-free image.
    """
    
    ROI = get_ROI(image, threshold=ROI_threshold)
    max_vertex_idx = image[ROI].argmax()
    contrast_brain = image[ROI].mean()
    
    # run many instances of noise
    noise_instances = []
    max_values = []
    
    for kk in range(n_noise_instances):
        meas_noise = np.random.normal(0, perc_noise, len(y))  #np.random.normal(0, perc_noise, len(y)) #* perc_noise #  gaussian noise 
        y_noise = y + meas_noise
        
        try:
            image_noise = W.values @ y_noise.values
        except:
            image_noise = W @ y_noise.values

        noise_brain = image_noise[:n_brain]
        
        if SB:
            noise_brain = G['G_brain'].values.T @ noise_brain
        
        noise_instances.append(np.mean(noise_brain[ROI]))
        max_values.append(noise_brain[max_vertex_idx])
        
    noise_std = np.std(np.asarray(noise_instances))
    noise_mean = np.mean(np.asarray(noise_instances))
    
    CNR = contrast_brain/noise_std
    
    return CNR, contrast_brain, noise_mean, max_values, noise_instances, image.max()

def get_CNR_v2(image, image_cov):
    """
    Compute CNR using analytical noise estimate from image covariance.
    
    Alternative CNR computation that uses pre-computed image covariance
    matrix instead of Monte Carlo simulation.
    
    Parameters
    ----------
    image : numpy.ndarray or xarray.DataArray
        Reconstructed image values.
    image_cov : numpy.ndarray or xarray.DataArray
        Diagonal of image covariance matrix (variance at each vertex).
        
    Returns
    -------
    CNR : float
        Contrast-to-noise ratio at peak vertex (max_amplitude / sqrt(variance)).
    """
    
    max_vertex_idx = image.argmax()
    contrast = image.max()
    
    noise = image_cov[max_vertex_idx]
    CNR = contrast/noise
    
    return CNR

def get_localization_error(origin, image, head, ROI_threshold=0.01):
    """
    Compute localization error between true and reconstructed activation centers.
    
    Measures the Euclidean distance between the known activation location
    (origin) and the amplitude-weighted centroid of the reconstructed image.
    
    Parameters
    ----------
    origin : numpy.ndarray or xarray.DataArray
        3D coordinates of true activation center.
    image : numpy.ndarray or xarray.DataArray
        Reconstructed image values.
    head : Head model object
        Contains mesh vertex positions.
    ROI_threshold : float, optional
        Fraction of max for ROI definition (default: 0.01).
        
    Returns
    -------
    localization_error : float
        Distance in mm between true origin and reconstructed centroid.
    """
    
    ROI = get_ROI(image, threshold=ROI_threshold)  
    
    ## find the centroid ##    
    centroid = get_image_centroid(image, ROI, head)
    
    
    origin = np.reshape(origin.values, [1, 3])
    
    ## get distance between centroid and the origin
    localization_error = np.sum((origin - centroid)**2)**0.5
    
    return localization_error

def get_contiguous_blob(max_vertex, head, ROI):
    """
    Extract spatially contiguous region using depth-first search.
    
    Starting from the specified vertex, explores connected vertices within
    the ROI using mesh topology to identify a single contiguous region.
    
    Parameters
    ----------
    max_vertex : int
        Starting vertex index (typically the peak amplitude vertex).
    head : Head model object
        Contains mesh faces for determining vertex connectivity.
    ROI : numpy.ndarray or list
        Vertex indices defining the region to search within.
        
    Returns
    -------
    ROI_contig : list
        Vertex indices forming the contiguous region connected to max_vertex.
        
    Notes
    -----
    Uses depth-first search (DFS) algorithm. Two vertices are considered
    connected if they share a face in the mesh.
    """
    # get contiuguous region containing the max vertex
    
    stack = [max_vertex]
    visited = set()
    ROI_contig = []
    
    # Step 4: Perform DFS within the ROI
    while stack:
        v = stack.pop()
        if v not in visited:
            visited.add(v)
            ROI_contig.append(v)
            
            # Find all neighboring vertices in faces that contain `v`
            for face in head.brain.mesh.faces:
                if v in face:
                    for neighbor in face:
                        if neighbor in ROI and neighbor not in visited:
                            stack.append(neighbor)

    return ROI_contig


def get_percent_reconstructed(X_brain, X_scalp, y_orig, A):
    """
    Compute percentage of signal reconstructed in brain vs scalp layers.
    
    Projects reconstructed brain and scalp images back to measurement space
    and calculates what fraction of the total original signal is accounted
    for by each tissue layer.
    
    Parameters
    ----------
    X_brain : numpy.ndarray
        Reconstructed image values for brain vertices.
    X_scalp : numpy.ndarray
        Reconstructed image values for scalp vertices.
    y_orig : numpy.ndarray
        Original measurement vector.
    A : xarray.DataArray
        Forward model sensitivity matrix with 'is_brain' coordinate.
        
    Returns
    -------
    perc_brain : float
        Percentage of total signal reconstructed in brain (0-1).
    perc_scalp : float
        Percentage of total signal reconstructed in scalp (0-1).
        
    Notes
    -----
    NaN values are replaced with 1e-18 to avoid errors.
    """
    
    if np.isnan(X_brain).sum() > 0:
        X_brain = np.nan_to_num(X_brain, nan=1e-18)
        X_scalp = np.nan_to_num(X_scalp, nan=1e-18)
        
    y_brain = A[:, A.is_brain.values].values @ X_brain
    y_scalp = A[:, ~A.is_brain.values].values @ X_scalp
    
    perc_brain = y_brain.sum()/y_orig.sum()
    perc_scalp = y_scalp.sum()/y_orig.sum()
    
    return perc_brain, perc_scalp
    
    
    
def get_ROI_volume(head, ROI):
    """
    Compute enclosed volume of region of interest.
    
    Calculates volume enclosed by tetrahedral elements formed by ROI faces
    using the divergence theorem (sum of signed tetrahedron volumes).
    
    Parameters
    ----------
    head : Head model object
        Contains mesh faces and vertices.
    ROI : numpy.ndarray or list
        Vertex indices defining the region of interest.
        
    Returns
    -------
    volume : float
        Absolute volume in mm³ enclosed by ROI.
        
    Notes
    -----
    Includes all faces that have at least one vertex in ROI.
    """

    total_volume = 0
    faces = head.brain.mesh.faces
    vertices = head.brain.mesh.vertices
    ROI_faces = faces[np.any(np.isin(faces, ROI), axis=1)]

    for tri in ROI_faces:
        v0, v1, v2 = vertices[tri]
        total_volume += np.dot(v0, np.cross(v1,v2))/6
        
    return abs(total_volume)


def triangle_area(v0, v1, v2):
    """
    Compute area of a triangle using cross product.
    
    Parameters
    ----------
    v0, v1, v2 : numpy.ndarray
        3D coordinates of triangle vertices.
        
    Returns
    -------
    area : float
        Triangle area in mm².
    """
    return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

def get_ROI_surface_area(head, ROI, min_vertices_in_face=2):
    """
    Compute surface area of region of interest.
    
    Sums areas of all triangular faces that have at least min_vertices_in_face
    vertices within the ROI.
    
    Parameters
    ----------
    head : Head model object
        Contains mesh faces and vertices.
    ROI : numpy.ndarray or list
        Vertex indices defining the region of interest.
    min_vertices_in_face : int, optional
        Minimum number of vertices per face required for inclusion (default: 2).
        
    Returns
    -------
    area : float
        Total surface area in mm² of faces meeting the vertex criterion.
    """
    faces = head.brain.mesh.faces
    vertices = head.brain.mesh.vertices
    roi_vertex_set = set(ROI)
    area = 0.0
    for f in faces:
        count = sum(v in roi_vertex_set for v in f)
        if count >= min_vertices_in_face:
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            area += triangle_area(v0, v1, v2)
    return area


def get_total_weighted_surface_area(head, image):
    """
    Compute amplitude-weighted average surface area.
    
    Weights each triangular face by the mean amplitude of its three vertices,
    then computes the weighted average area across all faces.
    
    Parameters
    ----------
    head : Head model object
        Contains mesh faces and vertices.
    image : numpy.ndarray or xarray.DataArray
        Image amplitude values at each vertex.
        
    Returns
    -------
    weighted_area : float
        Weighted average surface area in mm² (sum of weighted areas / sum of weights).
        
    Notes
    -----
    Each triangle's weight is the mean of its three vertex amplitudes.
    """
    
    faces = head.brain.mesh.faces
    vertices = head.brain.mesh.vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Compute triangle areas using cross product
    cross_prod = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    # Mean weight per triangle
    mean_weights = (image[faces[:, 0]] +
                    image[faces[:, 1]] +
                    image[faces[:, 2]]) / 3.0

    # Weighted surface area
    weighted_area = np.sum(areas * mean_weights)
    total_area = np.sum(mean_weights)
    return weighted_area/total_area
        

def get_total_surface_area(head):
    """
    Compute total surface area of the mesh.
    
    Calculates the sum of all triangle areas in the mesh by calling
    get_total_weighted_surface_area with uniform weights.
    
    Parameters
    ----------
    head : Head model object
        Contains mesh faces and vertices.
        
    Returns
    -------
    total_area : float
        Total surface area in mm² of the entire mesh.
        
    Notes
    -----
    This is a wrapper that calls get_total_weighted_surface_area with
    weights of 1.0 for all vertices.
    """
    faces = head.brain.mesh.faces
    vertices = head.brain.mesh.vertices
    return get_total_weighted_surface_area(vertices, faces, np.ones(len(vertices)))
    
    
    
