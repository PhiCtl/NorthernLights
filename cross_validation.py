# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""

import matplotlib.pyplot as plt
import numpy as np

from data_preprocessing import split_data
from implementations import *
from utils import build_poly,compute_loss,accuracy,predict_labels,compute_RMSE



def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
    

def cross_validation(y, x, k_indices, k,degree,method=1,lambda_=0,gamma=0.1, max_iters=100,batch_size=1,loss_type = 'MSE',L1= False,seed=1):
    
    #Est ce que je met une valeur par defaut ?
    
    """return the loss of ridge regression."""
 
    # get k'th subgroup in test, others in train: TODO
   
    tr_ind=k_indices[k]
    te_ind=k_indices[k_indices!=tr_ind]
    
    x_tr=x[tr_ind]
    x_te=x[te_ind]
        
    y_tr=y[tr_ind]
    y_te =y[te_ind]
    
    
    # form data with polynomial degree: TODO
    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)
 
    # Calcul of the weight 
    initial_w = np.zeros((tx_tr.shape[1],))
    w=choose_your_methods(method, y_tr, tx_tr, lambda_, gamma, max_iters, batch_size)
    
    # calculate the loss for train and test data: TODO
    if method==5 or method==6:
        loss_tr=compute_loss(y_tr, tx_tr, w,loss_type,lambda_,L1)
        loss_te=compute_loss(y_tr, tx_tr, w,loss_type,lambda_,L1)
    
    else:
        loss_tr=compute_loss(y_tr, tx_tr, w,loss_type,0,L1)
        loss_te=compute_loss(y_tr, tx_tr, w,loss_type,0,L1)
    
    #Compute accuracy of the model
    accuracy_te=accuracy(y_te, tx_te,w)
    
    # ***************************************************

    return loss_tr, loss_te, accuracy_te




def cross_validation_mean_fold(y, x, k_indices, k_fold,degree, method, lambda_, gamma, max_iters,batch_size,loss_type,L1,seed):
    
    # define lists to store the loss of training data and test data
    losses_tr=[]
    losses_te=[]

    accuracy_te=[]
    variances = []
    
    for j in range(k_fold):
        #temporary lists for test and training losses
        loss_tr,loss_te,acc_te=cross_validation(y, x, k_indices, j,degree, method, lambda_, gamma, max_iters,batch_size,loss_type,L1,seed)
        
        losses_tr.append(loss_tr)
        losses_te.append(loss_te)
        
        accuracy_te.append(acc_te) 
        
        variances.append(loss_te)
        
        
    #mean of the loss and accuracy on the k folds  
    
    mean_accuracy_te = np.mean(accuracy_te)
    mean_loss_tr = np.mean(losses_tr)
    mean_loss_te  = np.mean(losses_te)
    
    #if loss went too high (->inf or nan) because of divergence, ignore it
    mean_loss_tr = np.ma.masked_array(mean_loss_tr, np.isnan(mean_loss_tr))
    mean_loss_te = np.ma.masked_array(mean_loss_te, np.isnan(mean_loss_te))
    
    return mean_loss_tr,mean_loss_te,mean_accuracy_te



def best_lambda_cross_validation(y, x, k_indices,k_fold,degree, method, lambdas, gamma, max_iters,batch_size,loss_type,L1,seed):
    """Returns best lambda for a given degree (based on smallest RMSE) and associated RMSE loss"""
    #lambda range
    #lambdas = np.logspace(-4, 0, 30)
    
    # split data in k fold
    #k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    losses_tr =np.zeros(len(lambdas))
    losses_te = np.zeros(len(lambdas))
    
    accuracy_te = np.zeros(len(lambdas))

    # cross validation: loop for each lambda on the k folds
    for ind_lam, lambda_ in enumerate(lambdas):
        
        #temporary lists for test and training losses
        loss_tr,loss_te,acc_te=cross_validation_mean_fold(y, x, k_indices, k_fold,degree, method,lambda_,  gamma, max_iters,batch_size,loss_type,L1,seed)
        
        losses_tr[ind_lam]=loss_tr
        losses_te[ind_lam]=loss_te
            
        
        accuracy_te[ind_lam]=acc_te
    plt.boxplot(loss_te)    
        
    #find optimal lambda
    
    best_lambda_ = np.unravel_index(np.argmin(losses_te), losses_te.shape)
    print("The best lambda is: ", lambdas[best_lambda_])
   
    #plot RMSE variance for each lambda
    #fig, ax1 = plt.subplots()
    #ax1.set_title('RMSE test data')
    #ax1.boxplot(loss_te_plot.T)  
    
    cross_validation_visualization(lambdas, losses_tr, losses_te)
    
    return best_lambda,accuracy_te[np.argmin(losses_te)],losses_te[np.argmin(losses_te)]




def best_degree_cross_validation(y, x, k_indices,k_fold,degrees, method, lambda_, gamma, max_iters,batch_size,loss_type,L1,seed):
    """Returns best lambda for a given degree (based on smallest RMSE) and associated RMSE loss"""
    #degree range
    #degrees = np.arange(2, 3)
    
    # split data in k fold
    #k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    losses_tr =np.zeros(len(degrees))
    losses_te = np.zeros(len(degrees))
    
    
    accuracy_te = np.zeros(len(degrees))

    # cross validation: loop for each lambda on the k folds
    for ind_deg,deg in enumerate(degrees):
        
        #temporary lists for test and training losses
        loss_tr,loss_te,acc_te=cross_validation_mean_fold(y, x, k_indices, k_fold,deg,method , lambda_, gamma, max_iters,batch_size,loss_type,L1,seed)
        
        losses_tr[ind_deg]=loss_tr
        losses_te[ind_deg]=loss_te
            
        accuracy_te[ind_deg]=acc_te
        
        
    #find optimal degree
    
    best_degree = np.unravel_index(np.argmin(losses_te), losses_te.shape)
    print("The best degree is: ", degrees[best_degree])
   
    
    
    cross_validation_visualization(degrees, losses_tr, losses_te)
    
    return best_degree,accuracy_te[np.argmin(losses_te)],losses_te[np.argmin(losses_te)]
    
def best_parameters_cross_validation(y, x, k_indices, k_fold,degrees,method, lambdas, gamma,max_iters,batch_size,loss_type,L1,seed):
    
    # define lists to store the loss of training data and test data
    losses_tr =np.zeros((len(degrees), len(lambdas)))
    losses_te = np.zeros((len(degrees), len(lambdas)))
    

    accuracy_te = np.zeros((len(degrees), len(lambdas)))


    # cross validation over different degrees and for different lambdas in k folds
    for ind_deg,deg in enumerate (degrees):
        for ind_lam,lambda_ in enumerate(lambdas):
            #temporary lists for test and training losses
            loss_tr,loss_te,acc_te=cross_validation_mean_fold(y, x, k_indices, k_fold,  deg,method,lambda_, gamma, max_iters,batch_size,loss_type,L1,seed)
            
            losses_tr[ind_deg,ind_lam]=loss_tr
            losses_te[ind_deg,ind_lam]=loss_te
            
         
            accuracy_te[ind_deg,ind_lam]=acc_te
            
    #visualization
            
    cross_validation_visualization(lambdas,losses_tr[0, :],losses_te[0, :])       
    cross_validation_visualization(degrees, losses_tr[:, 0],losses_te[:, 0])
    
    
    #unravel_index return de position of np.argmin(losses_te) in the table of shape "losses_te.shape"
    best_value = np.unravel_index(np.argmin(losses_te), losses_te.shape)
            
    print("Best degree: %d, with lambda %.2E " %(degrees[best_value[0]],lambdas[best_value[1]]))
    
    return best_value,accuracy_te[np.argmin(losses_te)],losses_te[np.argmin(losses_te)]



def choose_your_methods(method, y_tr, tx_tr, lambda_, gamma, max_iters, batch_size):
    
   # create initial w for methods using it
        initial_w = np.zeros(tx_tr.shape[1])

        if method == 1:
            # Use least squares method
            w,_= least_squares(y_tr,tx_tr)
            return w
            
        if method == 2:
            # Use least squares GD
            w,_ = least_squares_GD(y_tr, tx_tr, initial_w,max_iters,gamma)
            return w
            
        if method == 3:
            # Use least squares SGD
            w,_ = least_squares_SGD(y_tr, tx_tr, initial_w, batch_size, max_iters, gamma)
            return w
            
        if method == 4:
            # Use ridge regression
            w=ridge_regression(y_tr, tx_tr, lambda_)
            return w
            
        if method == 5:
            # Use logistic regression
            w,_ = logistic_regression(y_tr, tx_tr, initial_w, max_iters, gamma)
            return w
            
        if method == 6:
            # Use regularized logistic regression
            w,_ = reg_logistic_regression(y_tr, tx_tr,initial_w, gamma, max_iters, lambda_)
            return w
            
    
    






