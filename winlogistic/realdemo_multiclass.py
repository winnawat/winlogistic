"""
Demo file on real-world data.
Multiclass one-vs-one classification on digits dataset.
Runs logistic regression from winlogistic on the spam dataset.
Visualizes the preprocess
regularization constant, lambda, is 1.
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
import warnings

from winlogistic import winlogistic as w

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# simulating data with 2 classes
X_master, y_master = w.get_data(source='digits')

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

# run multiclass one-vs-one classification
y_pred = w.multiclass_ovo(X_train, X_test, y_train,
                          beta_init, theta_init, lambduh, stepsize)

# compare prediction against test set
multiclass_me = np.mean(y_pred != y_test)
print("multiclass misclassification error is %f" % multiclass_me)
