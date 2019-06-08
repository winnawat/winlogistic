# winlogistic
Polished code of implementation of L2 regularized logistic regression from scratch

---
This module contains implementation of L2 regularized logistic regression from scratch.  
Functions available are:  
```python
get_data(source='spam')
"Fetchs data from either ESL's spam or sklearn's digits"

sim_data(n=100, dim=30, classes=2, sep=1, sd=1)
"Simulate data through normal distribution"

picktwoclasses(x, y, class1, class2)
"Selects two classes from X and Y array."

sigmoid(h)
"Computes sigmoid function h(x) = 1/(1+exp(-h))"

computegrad(x, y, beta, lambduh)
"Computes gradient of a logistic loss function"

objective(x, y, beta, lambduh)
"Computes objective value of a logistic loss dunction"

getstepsize(x, lambduh)
"Generate stepsize given x."

backtracking(X, y, stepsize, beta, lambduh, max_iter=100)
"Implements backtracking on the stepsize"

misclassificationerror(x, y, beta)
"Computes misclassification error of a binary logistic classifier"

fastgradalgo(x, y, beta, theta, lambduh, stepsize, accuracy=1e-4)
"Computes minimum through fast gradient descent with backtracking."

visualize(x_train, y_train, x_test, y_test, lambduh, iterations, beta_vals)
"Visualizes the training process in terms of objective value and misclassification error at a given iteration count"
crossval(x, y, lam, k=5)
"Finds the best value of lambda through cross validation"

multiclass_ovo(x, x_test, y, beta, theta, lambduh, stepsize, accuracy=1e-4)
"Peroforms multi-class classification using one-vs-ove logistc classifier."
```

---
## Instructions
Download the directory containing the code and import.
```python
# this imports all the functions in the file winlogistic.py from the directory winlogistic
import winlogistic as w

# example: fetching the spam dataset
x, y = w.get_data()
```
Or execute demo files on command line interface.
```
python simdemo.py
python realdemo.py
```

---
### Demo files
There are a few .py files for demonstration.
- simdemo.py
- realdemo.py
- realdemo_cv.py
- sklbenchmark.py
- realdemo_multiclass.py

All the files are executed in ipython notebook "polish-code-assignment.ipynb"
