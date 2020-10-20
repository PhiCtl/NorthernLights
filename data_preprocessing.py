# -*- coding: utf-8 -*-
import numpy as np

#--------------------------------GENERATE TRAIN AND TEST SET------------------------------------------------------------------#
"Generates train and test data from initial dataset"

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    #setting seed ensures reproducibility
    np.random.seed(seed)
    
    #shuffle vectors x and y
    shuffle_indices = np.random.permutation(len(y))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    
    #splitting data
    split_i = int(len(y) *ratio)
    
    y_training = shuffled_y[:split_i]
    y_test = shuffled_y[split_i:]
    x_training = shuffled_x[:split_i]
    x_test = shuffled_x[split_i:]
   
    return y_training, x_training, y_test, x_test


#------------------------------------------CATEGORICAL EXTRACTION----------------------------------------------------------------------#

def extract_jet(x, y, categ_ind, jet_num, indices):
    """Extract data for which PRI_jet_num = jet_num and remove undefined features"""
    
    #data points and corresponding output for which jet = jet_num
    jet_ind = np.nonzero(x[:,categ_ind] == jet_num)
    data_points = x[jet_ind] 
    output = y[jet_ind]
    #remove undefined features (indices) for those datapoints
    if(indices):
        data_points = np.delete(data_points, indices, 1)
    return data_points, output, jet_ind

def get_jets(x, y, cat_id, undefined_indices, list_ = False):
    
    """Extract 4 different data subsets depending on value of PRI_jet_num"""
    
    jet_0, y_0, i0 = extract_jet(x, y, cat_id, 0, undefined_indices[0])
    jet_1, y_1, i1 = extract_jet(x, y, cat_id, 1, undefined_indices[1])
    jet_2, y_2, i2 = extract_jet(x, y, cat_id, 2, undefined_indices[2])
    jet_3, y_3, i3 = extract_jet(x, y, cat_id, 3, undefined_indices[3])
    
    ind = [i0, i1, i2, i3]
    test = jet_0.shape[0] + jet_1.shape[0] + jet_2.shape[0] + jet_3.shape[0]
    test2 = len(y_0) + len(y_1) + len(y_2) + len(y_3)
    
    print("jet tot size:",test, " y tot size: ", test2)
    
    if list_:
        return [jet_0, jet_1, jet_2, jet_3], [y_0, y_1, y_2, y_3], ind
    else:   
        return jet_0, y_0, jet_1, y_1, jet_2, y_2, jet_3, y_3, ind

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
        #med_val = x_med[ind[1]]
        x[ind[0],ind[1]] = 0
    
    std =  (x - x_med) / x_mean
    
    return std



#-------------------------------------CORRELATION------------------------------------------------------------------------------#

"Correlation function -> returns x with deleted highly correlated features"


def correlation(x):
    "Calculate correlation and delete one of the 2 features if abs(correlation) is above 0.95"
    #calculate correlation between different features, r is the absolute value of correlation matrix
    r = abs(np.corrcoef(x.T)) 
    #Find index where abs(r) is close to 1
    index=np.argwhere(r >=0.95)
    #remove redundant data
    n=int(x.shape[1]/2)
    index=index[index[:,0]<=n+1]
    
    #delete one of the 2 features strongly correlated
    deleted_column=[]
    
    for i in range(0,index.shape[0]-2):
        if index[i,0] not in deleted_column and index[i,0]!=index[i,1]:
            x=np.delete(x, index[i,0],1)
            deleted_column=np.append(deleted_column, index[i,0])
    return x