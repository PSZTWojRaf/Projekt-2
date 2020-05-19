import pandas as pd
import xlrd
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# program ma na celu wizualizację idei naszego zadania

# przygotowanie danych
y = np.zeros((1,8))
for i in range(5):
        y = np.concatenate((y, np.ones((1,8))*(i+1)), axis=1 )

X = y;
np.size(X)
X = X+np.random.normal(0,0.3,np.size(X))

y = np.ravel(y)
X = np.transpose(X)

# funkcja celu na podstawie:
# https://scikit-learn.org/stable/modules/svm.html#svm-regression
# punkt 1.4.7.2 LinearSVC
def obj_func(w):
    penalty = 0
    epsilon = 0
    C = 100
    for i in range(np.size(X)):
        penalty = penalty + max((0 , np.linalg.norm(y[i]-(w[0]*X[i]+w[1]),ord = 1) - epsilon ))

    return 0.5 * np.power(np.linalg.norm(w[:-1], ord = 2), 2) + C * penalty

# i już nasza standardowa funkcja do minimalizacji
import scipy.optimize as opt
from scipy.optimize import BFGS

w = [0,0]
res = opt.minimize(obj_func, w, method='trust-constr', jac='2-point', hess=BFGS(), options={'verbose': 1})
print(res.x)

# dla porównania to samo z gotowej biblioteki
from sklearn import svm

regr = svm.LinearSVR()
regr.fit(X,y)

# wizualizacja
plt.figure()
plt.scatter(X, y, c=y, cmap='summer')
z = np.linspace(-1, 6, 20)

# "moja" regresja
plt.plot(z,res.x[0]*z+res.x[1])
# regresja z biblioteki
plt.plot(z,regr.predict(z.reshape(-1, 1)), linestyle='dashed')
plt.show()
