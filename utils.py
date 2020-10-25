# -*- coding: utf-8 -*-

import numpy as np

#-------------------------------------------------LOSSES-------------------------------------------------------------#

def compute_loss_MSE(y, tx, w, L1_reg, lambda_):
    """Calculate the MSE loss (with L2 regularization if lambda is not 0)"""
    e = y - tx.dot(w)
    if L1_reg:
        return 0.5*np.mean(e**2) + lambda_*np.linalg.norm(w,1)
    else:    
        return 0.5*np.mean(e**2) + lambda_*(np.linalg.norm(w)**2)


def compute_RMSE(y, tx, w, L1_reg, lambda_):
    "Calculate the RMSE loss"
    return np.sqrt(2*compute_loss_MSE(y, tx, w, L1_reg, lambda_))


def compute_loss_logREG(y, tx, w, lambda_):
    """compute the loss: negative log likelihood."""
    sig = sigmoid(tx.dot(w))
    loss = y.T.dot( np.log(sig) ) + (1 - y).T.dot( np.log(1-sig) )
    return -np.sum(loss) + np.squeeze(w.T.dot(w))*lambda_

def compute_loss_MAE(y, tx, w):
    """Calculate the loss using mae """
    e = y - tx @ w
    return (1/len(y) * np.sum(np.abs(e), axis = 0) )

def compute_loss(y, tx, w, loss_type = 'MSE', lbd = 0, L1 = False):
    """Compute loss for all"""
    
    
    if loss_type == 'RMSE':
        return compute_RMSE(y, tx, w, L1, lbd)
    if loss_type == 'MAE':
        return compute_loss_MAE(y, tx, w)
    if loss_type == 'logREG':
        return compute_loss_logREG(y, tx, w, lbd)  
    
    return compute_loss_MSE(y, tx, w, L1, lbd)

#---------------------------------------GRADIENT--------------------------------------------------------#

def compute_LS_gradient(y, tx, w):
    """Compute the gradient of Least squares GD."""

    e = y - tx.dot(w)
    N = len(e)
    return -1/N * tx.T.dot(e)


def calculate_gradient_logREG(y, tx, w, lambda_):
    """compute the gradient of loss."""

    sig=sigmoid(tx.dot(w))
    grad=tx.T.dot(sig-y) + 2*lambda_*w
    return grad

def compute_gradient(y, tx, w, method, lambda_ = 0, batch_s = 1):
    """Compute gradient"""
    #least squares SGD uses this gradient in a loop
    if method == 2:
        return compute_LS_gradient(y, tx, w)
  
    if method == 6:
        return calculate_gradient_logREG(y, tx, w, lambda_)
    else:
        print("Error: no method specified")

#----------------------------------------FEATURES AUGMENTATION ------------------------------------------------------------#


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.ones((len(x),1))
    for i in range(1, degree+1):
        phi = np.c_[phi, np.power(x,i)] 
        
    return phi

#------------------------------------------ACCURACY-------------------------------------------------------------#

def predict_labels(y_pred):
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred


def accuracy(y_true, y_pred):
    
    if(len(y_true) != len(y_pred)):
        print("Error: sizes don't match")
    else:
        y_pred = predict_labels(y_pred)
        acc = np.equal(y_true, y_pred)
        return np.sum(acc)/len(y_true)


#-----------------------------STOCHASTIC GRADIENT DESCENT-----------------------------------------------#
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
#-----------------------------LOGISTIC REGRESSION ------------------------------------------------------#


def sigmoid(t):
    """apply the sigmoid function on t."""
    ft=1/(1+np.exp(-t))
    return ft



def learning_by_gradient_descent(y, tx, w, gamma):
   
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
  
    # compute the loss: 
    loss=compute_loss(y, tx, w, 'logREG')
  
    # compute the gradient: 
    gradient=compute_gradient(y, tx, w, 6)

    # update w
    w=w-(gamma*gradient)

    return loss, w
#--------------------------------------------------------------------------------------------------------#
