import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize as opt


from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=250, centers=2,
                  random_state=0, cluster_std=0.55)
def obj_func(w):
    return np.power(np.linalg.norm(w[:-1], ord = 2), 2)

y[y==0]=-1

constr = np.concatenate((np.reshape(X[:,0]*y,(X.shape[0],1)),np.reshape(X[:,1]*y,(X.shape[0],1)),np.reshape(-y,(X.shape[0],1))) ,axis=1)
min = np.ones(X.shape[0])
max = np.ones(X.shape[0])*np.inf
#print(constr)

linear_constraint = opt.LinearConstraint(constr,min,max)

from scipy.optimize import BFGS
w = [10,10,10]
res = opt.minimize(obj_func, w, method='trust-constr', constraints=linear_constraint, jac='2-point', hess=BFGS(), options={'verbose': 1})

print(res.x)

width = 2/np.linalg.norm(res.x[:-1], ord = 2)

z = np.linspace(-1, 5, 1000)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')
plt.plot(z, res.x[2]/res.x[1]-res.x[0]/res.x[1]*z)
plt.plot(z, (res.x[2]/res.x[1]-res.x[0]/res.x[1]*z)+width/2,linestyle='dashed' )
plt.plot(z, (res.x[2]/res.x[1]-res.x[0]/res.x[1]*z)-width/2,linestyle='dashed' )
plt.show()
