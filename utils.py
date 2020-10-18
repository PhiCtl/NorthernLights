# -*- coding: utf-8 -*-
"""A function to compute the cost."""
import numpy as np

def compute_loss
#amodifier


def compute_gradient

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
#___________________


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.ones((len(x),1))
    for i in range(1, degree+1):
        phi = np.c_[phi, np.power(x,i)] 
        
    return phi

#------------------



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