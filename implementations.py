# -*- coding: utf-8 -*-
import numpy as np
from utils import compute_loss, compute_gradient
from data_preprocessing import *

#-----------------------------------lEAST SQUQRES-----------------------------------------------------#
def least_squares(y, tx):
    """calculate the least squares solution."""
    #Find w
    inv_=np.linalg.inv(np.dot(tx.T,tx)) 
    w_opt=np.dot(np.dot(inv_,tx.T),y.T)
    
    #compute loss
    e=(y-tx @ w_opt)**2
    mse=(1/len(y))*(sum(e))
    loss = np.sqrt(2 * mse)
    return w_opt,loss

#-----------------------------------LEAST SQUARES GRADIENT DESCENT--------------------------------------------------------------------#

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    """Least_squares method using gradient descent algorithm.
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

    return ws[-1], losses[-1]

#-----------------------------------LEAST SQUARES STOCHASTIC GRADIENT DESCENT--------------------------------------------------------------------#

def least_square_SGD(y, tx, iter_, batch_size,  gamma):
    "Least_squares method using stochastic gradient descent"
    
    # Building the initial model
    w =  np.zeros((tx.shape[1],1))
    losses=[]
    # Performing Gradient Descent
    for n in range(iter_):
        for y_b, tx_b in batch_iter(y, tx, batch_size, num_batches=1):
             #compute stochastic gradient and error
            grad=compute_gradient(y, tx, w, 2, batch_s = 1)
            #new w
            w = w - gamma * grad
            #loss
            losses.append(compute_loss(y_b, tx_b, w, 'RMSE'))
            
                
    return w,losses[-1]

#--------------------------------------------------RIDGE REGRESSION-----------------------------------------------------------------------#

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"
    Returns RMSE and w_opt"""
    
    #optimal weights calculated explicitly
    n = tx.shape[1]
    A = tx.T.dot(tx) + lambda_*2*n * np.identity(n)
    b = tx.T.dot(y)
    w_opt = np.linalg.solve(A,b)
    
    rmse = compute_loss(y, tx, w_opt, loss_type = 'RMSE', lbd = lambda_)
    
    #returns the root mean squared error associated with the optimal weights
    return w_opt, rmse


#--------------------------------------------------LOGISTIC REGRESSION-----------------------------------------------------------------------#


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    """logistic regression.
    Returns w_opt and sum_of_loss"""
    y[np.where(y == -1)] = 0 #change (-1,1) to (0,1)
    w=initial_w
    losses = []
    for n_iter in range(max_iters):
        grad= compute_gradient(y, tx, w, 6)
        loss = compute_loss(y, tx, w, 'logREG')
        losses.append(loss)
        w = w - (gamma * grad)
        
    return w, losses[-1]

#----------------------------------------REGULARIZED LOGISTIC REGRESSION---------------------#


def reg_logistic_regression(y, tx, w, gamma, max_iters, lambda_):
   
    """
    Penalized logistic regression using gradient descent algorithm.
    Return the loss and updated w.
    """
   
    losses=[]
    
    # Performing Gradient Descent
    for n_iter in range(max_iters):
        
        #Compute loss and gradient
        loss=compute_loss(y, tx, w, 'logREG',lambda_)
        gradient=compute_gradient(y, tx, w, 6, lambda_)
    
        #update w
        w=w-gamma*gradient
        
        
        if (n_iter % 100 == 0):
            # print average loss for the last print_every iterations
            print("Current iteration={i}, loss={l}".format(i=n_iter, l=loss))
    
        #Store losses   
        losses.append(loss)
        
        #Check convergence
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
           
    return w, loss

