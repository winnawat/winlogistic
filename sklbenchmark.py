"""
Benchmark file that executes both winlogistic and sklearn logistic classifier
and compares the result.
Dataset used is the spam dataset.
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings

import winlogistic as w

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# simulating data with 2 classes
X_master, y_master = w.get_data()

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

# obtain beta for scoring
fg_beta = fg[0]
winlogscore = 1 - w.misclassificationerror(X_test, y_test, fg_beta)

# initialize skleanr classifier
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)

# get sklearn score
sklscore = clf.score(X_test, y_test)

print("dataset used: spam dataset")
print("lambda for winlogistic is 1, C for sklearn is 1")
print("winlogistic's score on test set is %f" % winlogscore)
print("sklearn's score on test set is %f" % sklscore)
