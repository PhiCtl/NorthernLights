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
    
def accuracy_visualization(y1,y2,y3):
    

# Define data for chart
    x = np.arange(1,15,1)
    #y1 = acc_LS
    #y2 = acc_LR
    #y3 = acc_RegLogReg
    

# Set up figure
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)

# Plot Least_squares accuracy
    i1=ax.plot(x, y1,
            color = 'k',
            linestyle = '-',
            markersize = 8,
            marker = 'o',
            markerfacecolor='k',
            markeredgecolor='k',
            label  = 'Least_squares accuracy')



# Plot Ridge regression 
    i2=ax.plot(x, y2,
            color = 'b',
            linestyle = '-',
            markersize = 8,
            marker = 'o',
            markerfacecolor='b',
            markeredgecolor='b',
            label  = 'Ridge regression')


# Plot Regularized logistic regression 
    i3=ax.plot(x, y3,
            color = 'c',
            linestyle = '-',
            markersize = 8,
            marker = 'o',
            markerfacecolor='c',
            markeredgecolor='c',
            label  = 'Regularized logistic regression')

# Axes
    ax.grid(True, which='both')
    ax.set_xlabel('Degrees')
    ax.set_ylabel('Accuracy')
    ax.set_xscale("linear")

#Legend
    ax.legend()
    line_labels = ["Least squares", "Ridge regression", "Regularized Logistic regression"]

    ax.legend([i1, i2, i3],              # List of the line objects
           labels= line_labels,       # The labels for each line
           loc="center right",        # Position of the legend
           borderaxespad=0.1,         # Add little spacing around the legend box
           title="Legend Title")  
# Show plot
    plt.show()
    plt.savefig("fig/accuracy_plot")