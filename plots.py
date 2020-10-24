# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization(param, rmse_tr, rmse_te, param_type):
    """visualization the curves of loss_tr and loss_te."""
    
    #Param type can be: degree, lambda
    
    plt.semilogx(param, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(param, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel(param_type)
    plt.ylabel("Loss")
    plt.title("Cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("fig/cross validation for " + param_type)

