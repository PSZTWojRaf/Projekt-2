import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.optimize as opt

############################################

red_wine_df = pd.read_csv("winequality_dataset/winequality-red.csv", sep=";")
white_wine_df = pd.read_csv("winequality_dataset/winequality-white.csv", sep=";")

# Displaying first 5 rows of data
# print(red_wine_df.head())
# print(white_wine_df.head())

# Some examples of selecting and accessing data
# print(red_wine_df, '\n')
# print(red_wine_df['volatile acidity'], '\n')
# print(red_wine_df.keys(), '\n')
# print(red_wine_df.iloc[0],'\n')
# print(red_wine_df.iloc[:,0],'\n')

# Dividing dataset into training and validation set
msk = np.random.rand(len(red_wine_df)) < 0.8
red_wine_train = red_wine_df[msk]
red_wine_test = red_wine_df[~msk]

# print(red_wine_train.head())
# print(red_wine_test.head())

# Dividing dataset into training and validation set
msk = np.random.rand(len(white_wine_df)) < 0.8
white_wine_train = white_wine_df[msk]
white_wine_test = white_wine_df[~msk]

# print(white_wine_train.head())
# print(white_wine_test.head())

# print(min(red_wine_df.quality))
# print(max(red_wine_df.quality))
# print(min(white_wine_df.quality))
# print(max(white_wine_df.quality))

# print(white_wine_df.shape)

############################################


# plt.figure()
# plt.plot(np.arange(20))

# data = {'a': np.arange(50),
#         'c': np.random.randint(0, 50, 50),
#         'd': np.random.randn(50)}
# data['b'] = data['a'] + 10 * np.random.randn(50)
# data['d'] = np.abs(data['d']) * 100

# plt.figure()
# plt.scatter('a', 'b', c='c', s='d', data=data)
# plt.xlabel('entry a')
# plt.ylabel('entry b')

# data = {'a': np.arange(10),
#         'c': np.random.randint(0, 10, 10),
#         'd': np.random.randn(10)}
# data['b'] = data['a'] + 10 * np.random.randn(10)
# data['d'] = np.abs(data['d']) * 100
# data['q'] = np.random.randint(0, 10, 10)

# print(data)

# plt.figure()
# plt.scatter('a', 'b', c='c', s='q', data=data)
# plt.xlabel('entry a')
# plt.ylabel('entry b')



############################################

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


if dim == 2:
    plt.figure()
    plt.scatter(a[:,0], a[:,1])
    plt.scatter(b[:,0], b[:,1])
    plt.xlabel('entry a')
    plt.ylabel('entry b')
    # plt.show()

############################################

# SVM Implementation
x = [1,2,3]
euclidean_norm = np.linalg.norm(x, ord = 2)

print(euclidean_norm)

w = np.ones(dim+1)

w = [-1 3 5]

# print(w)

def obj_func(w):
    return np.power(np.linalg.norm(w[:-1], ord = 2), 2)

lower_bound = np.ones(2*n)
upper_bound = np.zeros(2*n) + np.inf

from scipy.optimize import BFGS
opt_lin_constr = opt.LinearConstraint([c[:,0]*c[:,2], c[:,1]*c[:,2], -c[:,2]], lower_bound, upper_bound)
#opt_lin_constr = opt.LinearConstraint(np.concatenate((np.transpose(np.matrix(c[:,0]*c[:,2])), np.transpose(np.matrix(c[:,1]*c[:,2])), np.transpose(np.matrix(-c[:,2]))), 1), lower_bound, upper_bound)

res = opt.minimize(obj_func, w, method='trust-constr', constraints=opt_lin_constr, jac='2-point', hess=BFGS(), options={'verbose': 1})

print(res)
