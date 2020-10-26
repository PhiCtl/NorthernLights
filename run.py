# -*- coding: utf-8 -*-

import numpy as np
from cross_validation import select_best_degree, select_best_lambda, choose_your_methods
from optimized_utils import accuracy, build_poly, predict_labels
from proj1_helpers import *
from data_preprocessing import *

#------------------------RUN FUNCTION---------------------------------------------------#

## CATEGORICAL EXTRACTION
undefined_features = [[4, 5, 6, 12, 22, 23, 24, 25, 26,
                       27, 28, 29], [4, 5, 6, 12, 22, 26, 27, 28], [22], [22]]
PRI_jet_num = 22

#USEFUL METHODS
   
#load data
y, tX, ids = load_csv_data('data/train.csv')
#train set and test set
y_tr, x_tr, y_te, x_te = split_data(tX, y, 0.8, seed=20)
    
#categorical extraction
    
jet_train, y_jet_train, index = get_jets(x_tr, y_tr, PRI_jet_num, undefined_features, list_ = True)
jet_test, y_jet_test, index_te = get_jets(x_te, y_te, PRI_jet_num, undefined_features, list_ = True)
    
#data preprocessing
jet_process_tr = []
jet_process_te = []
for jet in jet_train:
    jet_process_tr.append(preprocessing_data(jet, False, True, False))
for jet in jet_test:
    jet_process_te.append(preprocessing_data(jet, False, True, False))
    
best_degs = []
best_lbds = []
w_list = []
    
print("Data preprocessing: done")

#hyperparameter screening for best method (4: Ridge) by maximizing the accuracy on the k-folds
#we selected a special range for degrees and lambdas we came up with after several trials
for (jet, y_jet, index) in zip(jet_process_tr, y_jet_train, np.arange(4)):
    print("Jet n°:", index)
    best_degree, rmse_te, best_lambda, _ = select_best_parameter(y_jet, jet, 4, 'degree', by_accuracy = True, seed = 1 , k_fold = 5, degrees = np.array([1,5,6,7,8,10,12]), lambdas = np.array([0.009999, 0.1, 0.001]))
    best_degs.append(best_degree)
    best_lbds.append(best_lambda)
    
    #build the optimal weights
    x_augm = build_poly(jet, best_degree)
    w = best_w(y_jet , jet , 4, best_lambda , best_degree)
    w_list.append(w)
    
print("Grid search for ridge hyperparameters: done")
    
#compute accuracy on test set
y_pred_list = []
for(jet, w, deg) in zip(jet_process_te, w_list, best_degs):
    x_augm_t = build_poly(jet, deg)
    y_pred_ = x_augm_t.dot(w)
    y_pred_list.append(y_pred_)
    
y_predict = combine_jets(y_pred_list, index_te)
print("Ridge prediction for test set: ",accuracy(y_te, y_predict))
    
#generate submission file
    
#load data
_, tX_test, ids_test = load_csv_data('data/test.csv')
jet_final, ind_list = get_jets_final(tX_test, 22, undefined_features)
for jet in jet_final:
    jet = (preprocessing_data(jet, False, True, False))
    
final_jet = []
for (jet, w, deg) in zip(jet_final, w_list, best_degs):
    x_augm = build_poly(jet, deg )
    y_pred = x_augm.dot(w)
    final_jet.append(y_pred)
    
y_final = combine_jets(final_jet, ind_list)
Y = predict_labels(y_final)
    
create_csv_submission(ids_test, Y, 'predictions/best_prediction_ridge.csv')
print("Submission file created")
    
    