# -*- coding: utf-8 -*-
import numpy as np
from costs import compute_loss_MSE, compute_RMSE
from data_preprocessing import *

#-----------------------------------LEAST SQUARES GRADIENT DESCENT-------------------------------------------------------------------------#

" Useful to compute least_squares_GD "
def compute_gradient(y, tx, w):
    """Compute the gradient."""

    e = y - tx.dot(w)
    N = len(e)
    return -1/N * tx.T.dot(e)

"Least_squares method using gradient descent"

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    """linear regression using Gradient descent algorithm."""
   
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
   
    for n_iter in range(max_iters):
        
        #compute current gradient and loss(mse)
        w = ws[-1]
        g = compute_gradient(y, tx, w)
        loss = compute_RMSE(y, tx, w)
        #update w
        w = w - gamma*g
        
        # store w and loss(mse)
        ws.append(w)
        losses.append(loss)

    return losses[-1], ws[-1]

#--------------------------------------------------RIDGE REGRESSION-----------------------------------------------------------------------#

" Ridge regression using normal equations"

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    #optimal weights calculated explicitly
    n = tx.shape[1]
    A = tx.T.dot(tx) + lambda_*2*n * np.identity(n)
    b = tx.T.dot(y)
    w_opt = np.linalg.solve(A,b)
    
    rmse = compute_RMSE(y, tx, w_opt)
    
    #returns the root mean squared error associated with the optimal weights
    return rmse, w_opt