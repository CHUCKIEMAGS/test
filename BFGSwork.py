# Implementation of BFGS algorithm

import numpy as np 

X = np.array(([3,5], [5,1], [10,2]), dtype=float) # x data
y = np.array(([75], [82], [93]), dtype=float) # y data
degreeHyperparam = 3 # degrees of polynomial prediction
invHessian = np.identity(degreeHyperparam + 1) # initialize inverse Hessian

yhat = 60 * X # prediction initializer

errorSum = np.sum(np.square(y - yhat))
print(errorSum)

hesDim = 3 #Number of rows/columns in Hessian matrix
Hess = np.identity(3)
print(Hess)