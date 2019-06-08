"""
Demo file on simulated data.
Runs logistic regression from winlogistic on a simulated data.
Visualizes the preprocess
regularization constant, lambda, is 1.
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from winlogistic import winlogistic as w

# simulating data with 2 classes
X_master, y_master = w.sim_data()

# splitting into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_master, y_master,
                                                    random_state=0)

# scaling data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# initialize parameters
beta_init = np.zeros(X_train.shape[1])[:, np.newaxis]
theta_init = np.zeros(X_train.shape[1])[:, np.newaxis]
lambduh = 1
stepsize = w.getstepsize(X_train, lambduh)

# run fast gradient
fg = w.fastgradalgo(X_train, y_train, beta_init, theta_init, lambduh, stepsize)
ite = fg[1][-1]
train_me = fg[3]
print("Finished training model after %i iterations" % ite)
print("Misclassification error on training set is %f" % train_me)

# visualize the training process
w.visualize(X_train, y_train, X_test, y_test, lambduh, fg[1], fg[2])
