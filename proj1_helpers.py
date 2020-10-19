# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids



#-----------------------------GET JETS FOR FINAL TEST SET------------------------------------------------------------#

def extract_jet(x, categ_ind, jet_num, indices):
    """Extract data for which PRI_jet_num = jet_num and remove undefined features"""
    
    #data points for which jet = jet_num
    ind = np.nonzero(x[:,categ_ind] == jet_num)
    jet = x[ np.nonzero(x[:,categ_ind] == jet_num)] 
    #remove undefined features (indices) for those datapoints
    if(indices):
        jet = np.delete(jet, indices, 1)
    return jet, ind


def get_jets_final(x, cat_ind, undefined_features, list_ = True):
    
    """Get jets for final test set"""
    jet0, ind_0 = extract_jet(x, cat_ind, 0, undefined_features[0])
    jet1, ind_1 = extract_jet(x, cat_ind, 1, undefined_features[1])
    jet2, ind_2 = extract_jet(x, cat_ind, 2, undefined_features[2])
    jet3, ind_3 = extract_jet(x, cat_ind, 3, undefined_features[3])
    
    ind_list = [ind_0, ind_1, ind_2, ind_3]
    
    if list_:
        return [jet0, jet1, jet2, jet3], ind_list
    else:
        return jet0, jet1, jet2, jet3, ind_list

def combine_jets(y_jets, indices):
    
    #build y 
    n = 0
    for jet in y_jets:
        n += len(jet)
    y = np.zeros((n,)) #don't write (n,1) it doesn't like it... 
    
    #fill in y with corresponding y_jets in order
    for i, ind in enumerate(indices):
        y[ind] = y_jets[i]
    
    return y

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
