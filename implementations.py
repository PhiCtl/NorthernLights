# -*- coding: utf-8 -*-
import numpy as np
from utils import compute_loss, compute_gradient
from data_preprocessing import *

#-----------------------------------LEAST SQUARES GRADIENT DESCENT--------------------------------------------------------------------#


"Least_squares method using gradient descent"

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    """linear regression using Gradient descent algorithm.
    Returns RMSE and w_opt"""
   
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
   
    for n_iter in range(max_iters):
        
        #compute current gradient and loss rmse
        w = ws[-1]
        g = compute_gradient(y, tx, w, 2)
        loss = compute_loss(y, tx, w, loss_type = 'RMSE' )
        #update w
        w = w - gamma*g
        
        # store w and loss(mse)
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]

#--------------------------------------------------RIDGE REGRESSION-----------------------------------------------------------------------#

" Ridge regression using normal equations"

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    Returns RMSE and w_opt"""
    
    #optimal weights calculated explicitly
    n = tx.shape[1]
    A = tx.T.dot(tx) + lambda_*2*n * np.identity(n)
    b = tx.T.dot(y)
    w_opt = np.linalg.solve(A,b)
    
    rmse = compute_loss(y, tx, w_opt, loss_type = 'RMSE')
    
    #returns the root mean squared error associated with the optimal weights
    return rmse, w_opt