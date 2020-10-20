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



def cross_validation_jets(param, rmse_te_list, param_type):
    
    colours = ['b','r','g','m']
    jets = np.arange(4)
               
    for (rmse_te, col, jet_num) in zip(rmse_te_list, colours, jets):
        txt = 'test error' + jet_num
        plt.semilogx(param, rmse_te, marker=".", color= col, label=txt)
    
    plt.xlabel(param_type)
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    
    plt.savefig("cross validation all jets 1 method for " + param_type)


def plot_variance(rmse_te, param):
    fig, ax1 = plt.subplots()
    title = 'RMSE test data versus ' + param
    ax1.set_title(title)
    ax1.boxplot(rmse_te.T)