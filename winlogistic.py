"""
Submission for DATA558 polish code assignment
Main file for winlogistic containing core functions to get data,
train logistic regression classifier, and visualize the training process.
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn import preprocessing


def get_data(source='spam'):
    """
    fetchs data from either ESL's spam or sklearn's digits
    X: n*d matrix of examples and their features
    y: n*1 vector of labels {1,-1} for spam dataset, digits for digits dataset.
    returns X,y
    """
    if source is 'spam':
        spam = pd.read_csv('https://web.stanford.edu/~hastie/'
                           'ElemStatLearn/datasets/spam.data',
                           delimiter=' ', header=None)
        # rename the label column
        spam.rename(columns={57: 'spam'}, inplace=True)
        # convert 0's to -1's
        spam['spam'] = spam['spam'].map({0: -1, 1: 1})
        X = spam.drop('spam', axis=1)
        y = np.array(spam.spam)
        y = y.reshape(len(y), 1)
        return X, y
    elif source is 'digits':
        digits = load_digits(return_X_y=True)
        X = digits[0]
        y = digits[1]
        y = y.reshape(len(y), 1)
        return X, y
    else:
        raise Exception("source is either 'spam' or 'digits'")


def sim_data(n=100, dim=30, classes=2, sep=1, sd=1):
    """
    simulate data through normal distribution
    n: number of examples for each class. 100 by default.
    dim: number of features/dimensions. 30 by default.
    classes: number of classes. 2 by default.
    sep: the difference between mean value of the classes. 2 by default.
    sd: standard deviation. 1 be default.
    sim_data produces X, an n*d matrix of exmaples and their features, and
    y, an n*1 vector of the corresponding labels.
    returns X, y
    """
    if classes < 2:
        raise Exception("at least two classes required")
    if type(classes) != int:
        raise Exception("classes has to be an integer")
    if type(n) != int:
        raise Exception("n has to be an integer")
    if type(dim) != int:
        raise Exception("dim has to be an integer")

    X = np.random.normal(loc=0, scale=sd, size=(n, dim))
    y = np.full((n, 1), 1)
    for i in range(2, classes+1):
        next_X = np.random.normal(loc=(0+(i-1)*sep), scale=sd, size=(n, dim))
        next_y = np.full((n, 1), i)
        X = np.append(X, next_X, axis=0)
        y = np.append(y, next_y, axis=0)

    if classes == 2:
        y = np.where(y == 2, -1, y)

    return X, y


def picktwoclasses(x, y, class1, class2):
    """
    Selects two classes from X and Y array.
    Args:
        x: array of features and samples. Shape n*d
        y: array of labels. Shape n*1
        class1: One of the two classes meant to be selected.
        class2: The other class meant to be selected.
    The two classes are now 1 and -1 for class1 and class2 respectively.
    Return: array of n features, array of n labels.
    """
    bool_mask1 = y == class1
    bool_mask2 = y == class2
    bool_mask = bool_mask1 + bool_mask2

    y_subset = np.extract(bool_mask, y)
    y_subset = np.where(y_subset == class1, 'class1', y_subset)
    y_subset = np.where(y_subset == str(class2), -1, y_subset)
    y_subset = np.where(y_subset == 'class1', 1, y_subset)
    y_subset = y_subset.reshape(len(y_subset), 1)
    y_subset = y_subset.astype(int)
    x_subset = x[bool_mask.ravel(), :]
    return x_subset, y_subset


def sigmoid(h):
    '''
    Computes sigmoid function h(x) = 1/(1+exp(-h))
    This produces p, probability.
    Arg:
        h
    Returns:
        sigmoid of h
    '''
    return np.power(1 + np.exp(-h), -1)


def computegrad(x, y, beta, lambduh):
    """
    Computes gradient of a logistic loss function
    Args:
        x: n*d features
        y: n*1 labels
        beta: beta
        lambduh: lambda value
    Return: gradient
    """
    n = len(x)
    beta = beta.reshape(len(beta), 1)
    yx = y*x
    power = yx.dot(beta)
    grad = 1/n * np.sum(-yx * (1-sigmoid(power)), axis=0)[:, np.newaxis]
    + 2 * lambduh*beta
    return grad


def objective(x, y, beta, lambduh):
    """
    computes objective value of a logistic loss dunction
    Args:
        x: n*d features
        y: n*1 labels
        beta: beta
        lambduh: lambda value
    Return: the objective value
    """
    objval = 1/len(y) * np.sum(np.log(1 + np.exp(-y * x.dot(beta))))
    + lambduh * np.linalg.norm(beta)**2
    return objval


def getstepsize(x, lambduh):
    '''
    generate stepsize given x.
    return initial stepsize value
    '''
    # find initial stepsize
    L = max(np.linalg.eigvals(1/len(x) * x.T.dot(x))) + lambduh
    stepsize = 1/L
    return stepsize


def backtracking(X, y, stepsize, beta, lambduh, max_iter=100):
    '''
    Implements backtracking on the stepsize
    '''
    alpha = 0.5
    gamma = 0.8
    grad = computegrad(X, y, beta, lambduh)

    # iterate n = (gamma)n
    iter = 0
    while (objective(X, y, beta - stepsize * grad,
                     lambduh)
            > objective(X, y, beta, lambduh)
            - (alpha*stepsize * (np.linalg.norm(grad))**2) and i < max_iter):
        stepsize = gamma*stepsize
        iter += 1
    return stepsize


def misclassificationerror(x, y, beta):
    """
    computes misclassification error of a binary logistic classifier
    Args:
        x: n*d array of examples and features
        y: n*1 array of labels
        beta: the beta value
    Return: misclassification error
    """
    y_pred = sigmoid(x.dot(beta)) > 0.5
    y_pred = y_pred*2 - 1  # Convert to +/- 1
    return np.mean(y_pred != y)


def fastgradalgo(x, y, beta, theta, lambduh, stepsize, accuracy=1e-4):
    """
    Computes minimum through fast gradient descent with backtracking.
    Args:
        x: n*d matrix of examples and features
        y: n*1 vector of examples' labels
        beta: the initial value of beta
        theta: the initial value of theta
        lambduh: regularization constant
        stepsize: the initial stepsize
        accuracy: accuracy threshold
    return:
        beta: The vaue of beta at the minumum
        iterations: a list of number of iterations
        beta_vals: a list of values of beta at its respective iteration count
        train_me: misclassification error on training set
    """
    grad_theta = computegrad(x, y, theta, lambduh)
    grad_beta = computegrad(x, y, beta, lambduh)

    beta_vals = []
    ite = 0
    iterations = []

    while np.linalg.norm(grad_beta) > accuracy:
        stepsize = backtracking(x, y, stepsize, theta, lambduh)
        beta_new = theta - stepsize*grad_theta
        theta = beta_new + ite/(ite+3)*(beta_new-beta)

        # Store beta values for each iteration
        iterations.append(ite)
        beta_vals.append(beta)

        grad_theta = computegrad(x, y, theta, lambduh)
        grad_beta = computegrad(x, y, beta, lambduh)
        beta = beta_new
        ite += 1

    train_me = misclassificationerror(x, y, beta)
    train_me = round(train_me, 3)

    return [beta, iterations, beta_vals, train_me]


def visualize(x_train, y_train, x_test, y_test,
              lambduh, iterations, beta_vals):
    """
    Visualizes the training process in terms of objective value and
    misclassification error at a given iteration count
    """
    train_objective_vals = []
    train_me_vals = []
    test_objective_vals = []
    test_me_vals = []
    for beta in beta_vals:
        train_me = misclassificationerror(x_train, y_train, beta)
        test_me = misclassificationerror(x_test, y_test, beta)
        train_obj = objective(x_train, y_train, beta, lambduh)
        test_obj = objective(x_test, y_test, beta, lambduh)

        train_objective_vals.append(train_obj)
        train_me_vals.append(train_me)
        test_objective_vals.append(test_obj)
        test_me_vals.append(test_me)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(iterations, train_me_vals, color='red', label='training set')
    plt.plot(iterations, test_me_vals, label='validation set')
    plt.grid(True)
    plt.title('misclassification error against iteration')
    plt.ylabel('misclassification error')
    plt.xlabel('iterations')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(iterations, train_objective_vals, color='red',
             label='training set')
    plt.plot(iterations, test_objective_vals, label='validation set')
    plt.grid(True)
    plt.title('objective value against iteration')
    plt.ylabel('objective value')
    plt.xlabel('iterations')
    plt.legend()

    plt.show()


def crossval(x, y, lam, k=5):
    """
    Finds the best value of lambda through cross validation
    Args:
        x: n*d the matrix of examples and features
        y: n*1 vector of the labels of the exmaples
        lam: a list of values of lambda
        k: number of fols in k-fold cross validation
    Returns the best value of lambda (lowest misclassification error)
    """
    n = len(x)
    nbyk = int(n/k)
    me_set = []

    for i in lam:
        print("Attempting %i-fold cross validation with lambda = %f" % (k, i))
        mes_i = []
        for cv in range(1, (k+1)):
            X_train_cv = np.append(x[:(cv-1)*nbyk, :], x[cv*nbyk:, :], axis=0)
            y_train_cv = np.append(y[:(cv-1)*nbyk, :], y[cv*nbyk:, :], axis=0)
            X_test_cv = x[(cv-1)*nbyk:cv*nbyk, :]
            y_test_cv = y[(cv-1)*nbyk:cv*nbyk, :]

            # Initialize things
            beta_init = np.zeros(X_train_cv.shape[1])[:, np.newaxis]
            theta_init = np.zeros(X_train_cv.shape[1])[:, np.newaxis]
            lambduh = i
            stepsize = getstepsize(X_train_cv, lambduh)

            fg = fastgradalgo(X_train_cv, y_train_cv, beta_init, theta_init,
                              lambduh, stepsize)
            me = misclassificationerror(X_test_cv, y_test_cv, fg[0])
            mes_i.append(me)
        me_set.append(np.mean(mes_i))

    best_lambda_cv = lam[me_set.index(min(me_set))]
    print("Best value of lambda from cross validation is %f" % best_lambda_cv)
    return best_lambda_cv


def multiclass_ovo(x, x_test, y, beta, theta, lambduh,
                   stepsize, accuracy=1e-4):
    """
    Peroforms multi-class classification using one-vs-ove logistc classifier.
    A multiclass one-vs-one logistic classifier.
    Runs binary classifier for all possible pairs of labels.
    Final prediction is the label which is predicted
    most frequently for the example.
    Args:
        x: n*d matrix of examples and features
        y: n*1 vector of examples' labels
        beta: the initial value of beta
        theta: the initial value of theta
        lambduh: regularization constant
        stepsize: the initial stepsize
        accuracy: accuracy threshold
    """
    allclasses = np.unique(y)
    pairs = list(itertools.combinations(allclasses, 2))
    pred_grid = pd.DataFrame(columns=pairs)

    for pair in pairs:
        x_train, y_train = picktwoclasses(x, y, pair[0], pair[1])

        # standardizing x
        scaler_x = preprocessing.StandardScaler()
        x_train = scaler_x.fit_transform(x_train)

        # fitting fastgradalgo
        fg = fastgradalgo(x_train, y_train, beta, theta, lambduh,
                          stepsize, accuracy)
        beta = fg[0]
        # predictions
        y_pred = sigmoid(x_test.dot(beta)) > 0.5
        y_pred = y_pred*2 - 1  # Convert to +/- 1

        # convert to respective classes
        y_pred = np.where(y_pred == 1, pair[0], y_pred)
        y_pred = np.where(y_pred == -1, pair[1], y_pred)
        y_pred = y_pred.ravel()

        pred_grid[pair] = y_pred

    y_pred_master = pred_grid.mode(axis=1)
    y_pred_master = np.array(y_pred_master[0])
    y_pred_master = y_pred_master.astype(int)
    y_pred_master = y_pred_master.reshape(len(y_pred_master), 1)
    return y_pred_master
