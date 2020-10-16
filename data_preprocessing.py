# -*- coding: utf-8 -*-
import numpy as np

#------------------------------------------CATEGORICAL EXTRACTION----------------------------------------------------------------------#

def extract_jet(x, categ_ind, jet_num, indices):
    """Extract data for which PRI_jet_num = jet_num and remove undefined features"""
    
    #data points for which jet = jet_num
    data_points = x[ np.nonzero(x[:,categ_ind] == jet_num)] 
    #remove undefined features (indices) for those datapoints
    if(indices):
        data_points = np.delete(data_points, indices, 1)
    return data_points

def get_jets(x, cat_id, undefined_indices):
    
    """Extract 4 different data subsets depending on value of PRI_jet_num"""
    
    jet_0 = extract_jet(x, cat_id, 0, undefined_indices[0])
    jet_1 = extract_jet(x, cat_id, 1, undefined_indices[1])
    jet_2 = extract_jet(x, cat_id, 2, undefined_indices[2])
    jet_3 = extract_jet(x, cat_id, 3, undefined_indices[3])
        
    return jet_0, jet_1, jet_2, jet_3

#-------------------------------------DATA STANDARDIZATION------------------------------------------------------------------------------#

"Standardization function -> returns matrix of values with zero mean and standard deviation of 1"

def standard(x):
    """Returns a standardized matrix, replacing "-999" entries by median (sensitivity to outliers) of matrix"""
    x[x == -999] = np.nan
    
    #setting nan values to median over all datapoints for a specific feature: worked but reduced to 0 when standardizing
    nan_indices = np.argwhere(np.isnan(x))
    x_med = np.nanmedian(x,axis = 0)
    x_mean = np.nanmean(x, axis = 0)
    for ind in nan_indices:
        med_val = x_med[ind[1]]
        x[ind[0],ind[1]] = med_val
    
    std =  (x - np.nanmean(x, axis = 0)) / np.nanstd(x, axis = 0)
    
    return std

#--------------------------------------FEATURES AUGMENTATION-----------------------------------------------------------------------#

"Feature augmentation"

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.ones((len(x),1))
    for i in range(1, degree+1):
        phi = np.c_[phi, np.power(x,i)] 
        
    return phi




