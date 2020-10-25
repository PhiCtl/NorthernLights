# Higgs Boson Challenge

The  Higgs  boson  is  an  elementary  particle  whose existence was announced in 1964 and observed for the first time in 2013 at the Large Hadron Collider of CERN. During a high speed collision event between billions of particle, it is possible to identifythe Higg Boson thanks to its decay signature. Data of a collisionevent can be collected, and be used as a training set for machinelearning models to be able to predict if a Higg Boson was emitted or not during a collision, based on the parameters recorded. This project aims to find a machine learning model that will have the best accuracy in predicting if a Higgs Boson was emitted or not during a collision event.

You can check the report in the github repository to have an overview of the project and an insight into the steps that led to our prediction model.

Our best prediction was obtained with ridge regression and the model had an overall accuracy of 0.82446 (submission on AICrowd).

## Installation

1.You can download the code to run the models and data from the github repository [Github](https://github.com/PhiCtl/NorthernLights) into the data directory

```bash
Clone the repository: git clone https://github.com/PhiCtl/NorthernLights
```
2.Unzip the train.csv in the data directory. Once in the right directory, type the following line:

```bash
Unzip train.zip
```
## Running the program

To run the program, open a terminal on Jupyter Notebook, go in the project repository and type the following command:

```bash
python run.py
```

The best method (ridge regression) will run and a file 'best_prediction_ridge.csv' will be generated in the folder predictions which contains the prediction and can be uploaded on AICrowd to make a submission.

## Notations and parameters
 The different method are referred as the following in the code:
```bash
Methods mapping
1    Least Squares
2    Least Squares Gradient Descent
3    Least Squares Stochastic Gradient Descent
4    Ridge Regression
5    Logistic Regression
6   Regularized Logistic Regression
```
The parameters encountered in the different methods are the following.

```bash
Gamma=waiting factor
Lambda=regularization factor
Max_iters=maximum number of iterations
Batch_size=the size of the batch used for stochastic gradient descent
Degree=degree for feature expansion
```
If you want to use a specific method, open the file cross_validation.ipynb and run the different cells selecting the method thanks to its refering number. You can also tune the parameter yourself, the ipynb format makes it easy to follow.

