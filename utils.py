# -*- coding: utf-8 -*-

import numpy as np

#-------------------------------------------------LOSSES-------------------------------------------------------------#

def compute_loss_MSE(y, tx, w, lambda_ = 0, L1_reg = False):
    """Calculate the MSE loss
    """
    e = y - tx.dot(w)
    if L1_reg:
        return 0.5*np.mean(e**2) + lambda_*np.linalg.norm(w,1)
    else:    
        return 0.5*np.mean(e**2) + lambda_*(np.linalg.norm(w)**2)


def compute_RMSE(y, tx, w, lambda_ = 0):
    "Calculate the RMSE loss"
    return np.sqrt(2*compute_loss_MSE(y, tx, w, lambda_ = lambda_))


def compute_loss_logREG(y, tx, w, lambda_ = 0):
    """compute the loss: negative log likelihood."""
    L1 = y.T.dot(tx.dot(w))
    L2 = np.sum(np.log(np.ones(tx.shape[0]) + np.exp(tx.dot(w)))) 
    return L1 + L2 + lambda_*np.linalg.norm(w)**2

def compute_loss_MAE(y, tx, w, lambda_ = 0):
    """Calculate the loss.
    You can calculate the loss using mae.
    """
    e = y - tx @ w
    return (1/len(y) * np.sum(np.abs(e), axis = 0) )

def compute_loss(y, tx, w, loss_type = 'MSE', lbd = 0, L1 = False):
    
    if loss_type == 'RMSE':
        return compute_RMSE(y, tx, w, lambda_ = lbd)
    if loss_type == 'MAE':
        return compute_loss_MAE(y, tx, w, lambda_ = lbd)
    if loss_type == 'logREG':
        return compute_loss_logREG(y, tx, w, lambda_ = lbd)  
    
    return compute_loss_MSE(y, tx, w, lambda_ = lbd,  L1_reg = L1)

#---------------------------------------GRADIENT--------------------------------------------------------#

def compute_LS_gradient(y, tx, w):
    """Compute the gradient."""

    e = y - tx.dot(w)
    N = len(e)
    return -1/N * tx.T.dot(e)


def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    g = np.array([0, 0])
    
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        g = g + compute_gradient(minibatch_y, minibatch_tx, w)
    
    return 1/batch_size * g

def calculate_gradient_logREG(y, tx, w):
    """compute the gradient of loss."""
  
    Xw=tx.dot(w)
    #sigma=np.exp(Xw)/1+np.exp(Xw) Pk c est pas pareil que sigmoid ??
    sig=sigmoid(tx.dot(w))
    grad=tx.T.dot((sig)-y)
    
    return grad

def compute_gradient(y, tx, w, method, batch_s = 1):
    
    if method == 2:
        return compute_LS_gradient(y, tx, w)
    if method == 3:
        return compute_stoch_gradient(y, tx, w, batch_size = batch_s)
    if method == 6:
        return calculate_gradient_logREG(y, tx, w)
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

def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def predict_labels_2(y_pred):
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred

def accuracy(y_true, x_te, w_opt):
                 
    """Compute accuracy of a given model"""
    y_test = predict_labels(w_opt, x_te)
    acc = np.equal(y_test, y_true)
    return np.sum(acc)/len(y_test)

def accuracy_2(y_true, y_pred):
    
    if(len(y_true) != len(y_pred)):
        print("Error: sizes don't match")
    else:
        y_pred[y_pred <= 0] = -1
        y_pred[y_pred > 0] = 1
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


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    # calculate Hessian
    sig = sigmoid(tx.dot(w))
    sig = np.diag(sig.T[0])
    r = np.multiply(sig, (1-sig))
    return tx.T.dot(r).dot(tx)


def elements_logistic_regression(y, tx, w):
    """return the loss, gradient, and Hessian."""
 
    # return loss, gradient, and Hessian: TODO
    
    loss=calculate_loss_logREG(y, tx, w)
    gradient=calculate_gradient_logREG(y, tx, w)
    hessian=calculate_hessian(y, tx, w)
    return loss, gradient, hessian
    
def learning_by_gradient_descent(y, tx, w, gamma):
    #TODO: a completer avec max iter
   
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    # ***************************************************
    # compute the loss: 
    loss=calculate_loss_logREG(y, tx, w)
  
    # compute the gradient: 
    gradient=calculate_gradient_logREG(y, tx, w)
    # ***************************************************

    # update w
    w=w-(gamma*gradient)

    return loss, w



def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and Hessian."""
   
    # return loss, gradient, and Hessian
    
    loss,gradient,hessian=elements_logistic_regression(y,tx,w)
    loss=loss+lambda_*np.dot(w.T,w)
    gradient=gradient+2*lambda_*w
    hessian=hessian+2*lambda_
    
    return loss,gradient,hessian

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    # return loss, gradient: TODO
    loss,gradient,hessian=penalized_logistic_regression(y, tx, w, lambda_)
    # ***************************************************
    # update w
    w=w-gamma*gradient

    return loss, w