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

#-----------------------------------LEAST SQUARES STOCHASTIC GRADIENT DESCENT--------------------------------------------------------------------#


"Least_squares method using stochastic gradient descent"

def least_square_SGD(y, tx, iter_, batch_size,  gamma):
    
    """linear regression using stochastic Gradient descent algorithm.
    Returns w and losses"""
    
    # Building the initial model
    w,loss=least_squares(y,tx)
    losses=np.array([loss])
    # Performing Gradient Descent
    for n in range(iter_):
        for y_b, tx_b in batch_iter(y, tx, batch_size, num_batches=1):
             #compute stochastic gradient and error
            grad,e=compute_stoch_gradient(y_b, tx_b,w)
            #new w
            w = w - gamma * grad
            np.append(losses, y - tx.dot(w))
            
                
    return w,losses

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


#--------------------------------------------------LOGISTIC REGRESSION-----------------------------------------------------------------------#

"Logistic regression"

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    """logistic regression.
    Returns w_opt and sum_of_loss"""
    y[np.where(y == -1)] = 0 #change (-1,1) to (0,1)
    w=initial_w
    sum_loss=0
    for n_iter in range(max_iters):
        grad= compute_log_grad(y, tx, w)
        loss = compute_log_loss(y, tx, w)
        sum_loss += loss
        w = w - (gamma * grad)
        
    return w,sum_loss