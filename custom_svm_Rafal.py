import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize as opt


# Number of points
n = 12
# Dimension
dim = 2
# Distance betweend points sets
d = 10
# Range <0, r>
r = 10


# Python is smarter than me :/
# a = np.zeros([n, dim])
# b = np.zeros([n, dim])
# for i in range(n):
#     for j in range(dim):
#         a[i, j] = np.random.rand()*100%(r+1)

# for i in range(n):
#     for j in range(dim):
#         b[i, j] = np.random.rand()*100%(r+1) + d


a = np.random.rand(n, dim)*100%(r+1)
b = np.random.rand(n, dim)*100%(r+1) + d

a2 = np.concatenate((a,np.ones((n, 1))), 1)
b2 = np.concatenate((a,np.ones((n, 1))*-1), 1)

c = np.concatenate((a2, b2))

#print(c)


#if dim == 2:
#    plt.figure()
#    plt.scatter(a[:,0], a[:,1])
#    plt.scatter(b[:,0], b[:,1])
#    plt.xlabel('entry a')
#    plt.ylabel('entry b')
#    # plt.show()

def obj_func(w):
    return np.power(np.linalg.norm(w[:-1], ord = 2), 2)

X = np.concatenate((np.reshape(c[:,0]*c[:,-1],(c.shape[0],1)),np.reshape(c[:,1]*c[:,-1],(c.shape[0],1)),np.reshape(-c[:,-1],(c.shape[0],1))) ,axis=1)
min = np.ones(c.shape[0])
max = np.ones(c.shape[0])*np.inf

print(X)

linear_constraint = opt.LinearConstraint(X,min,max)

from scipy.optimize import BFGS
w = [1,1,1]
res = opt.minimize(obj_func, w, method='trust-constr', constraints=linear_constraint, jac='2-point', hess=BFGS(), options={'verbose': 1})

print(res.x)

z = np.linspace(0, 20, 1000)
plt.figure()
plt.scatter(a[:,0], a[:,1])
plt.scatter(b[:,0], b[:,1])
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.plot(z, w[2]/w[1]-w[0]/w[1]*z )
plt.show()
