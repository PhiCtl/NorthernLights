# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt


def cross_validation_visualization(param, rmse_tr, rmse_te, param_type):
    """visualization the curves of rmse_tr and rmse_te."""
    
    #Param type can be: degree, gamma, lambda
    
    plt.semilogx(param, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(param, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel(param_type)
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross validation for " + param_type)



def plot_variance(rmse_te, param):
    fig, ax1 = plt.subplots()
    title = 'RMSE test data versus ' + param
    ax1.set_title(title)
    ax1.boxplot(rmse_te.T)
    