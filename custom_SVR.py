# standard imports
import numpy as np
import math
import matplotlib.pyplot as plt

# library that contains quadprog optimization tool
# pip install cvxopt
from cvxopt import matrix
from cvxopt import solvers

# kernel functions
def K(x1,x2,type = 'linear',degree = 2, gamma = 1 ):
    if type == 'linear':
        return np.dot(x1,x2)
    elif type == 'poly':
        return math.pow(np.dot(x1,x2)+1, degree)
    elif type == 'rbf':
        return math.exp(-np.linalg.norm(x1-x2, 2)*gamma)

class SVR:
    def __init__(self,C=1, epsilon=0, kernel = 'linear', degree = 2, gamma = 1):
        # params
        self.C = C;
        self.epsilon = epsilon;
        self.kernel = kernel;
        self.degree = degree;
        self.gamma = gamma;

    def fit(self, X, y):
        # training data
        self.X = X;
        self.y = y;
        # number of samples
        n = np.size(self.X,0);
        self.n = n;

        # optimization process
        e = np.ones(n);
        Q = np.zeros((n,n))

        # matrix Q
        i = 0
        for x1 in X:
            j = 0
            for x2 in X:
                Q[i,j] = K(x1,x2,self.kernel,self.degree,self.gamma)
                j += 1
            i += 1

        # matrix H
        P = np.zeros((4*n,4*n))
        P[0:n,0:n] = Q

        # vector f
        q = np.concatenate((-y,e*self.epsilon,np.zeros(2*n)))

        # equality constraints

        # subject to: SUM(ai-ai*) = 0
        sub1 = np.concatenate( (e,np.zeros(3*n)), axis=0)
        # a-a* -a +a* = 0
        sub2 = np.concatenate(( np.eye(n), np.zeros((n,n)), -np.eye(n), np.eye(n) ), axis=1)
        # a+a* -a -a* = 0
        sub3 = np.concatenate(( np.zeros((n,n)), np.eye(n), -np.eye(n), -np.eye(n) ), axis=1)

        A = np.concatenate(([sub1], sub2, sub3), axis=0)
        # vector of 0
        b = np.zeros(np.size(A,0))

        # inequality constraints

        # a >= 0
        sub1 = np.concatenate( (np.zeros((n,n)), np.zeros((n,n)), -np.eye(n), np.zeros((n,n))) , axis=1)
        # a* >= 0
        sub2 = np.concatenate( (np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), -np.eye(n)) , axis=1)
        # a <= C
        sub3 = np.concatenate( (np.zeros((n,n)), np.zeros((n,n)), np.eye(n), np.zeros((n,n))) , axis=1)
        # a* <= C
        sub4 = np.concatenate( (np.zeros((n,n)), np.zeros((n,n)), np.zeros((n,n)), np.eye(n)) , axis=1)

        G = np.concatenate( (sub1, sub2, sub3, sub4), axis=0 )
        h = np.concatenate( (np.zeros(2*n), np.ones(2*n)*self.C) )

        # conversion to aproperiate form for cvxopt library
        P = matrix(P)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A)
        b = matrix(b)

        # quadratic programming
        sol = solvers.qp(P,q,G,h,A,b);

        # solutions to numpy arrays
        self.a_minus_a_star = np.array(sol['x'][0:n])
        self.a_plus_a_star =  np.array(sol['x'][n:2*n])
        self.a = np.array(sol['x'][2*n:3*n])
        self.a_star = np.array(sol['x'][3*n:4*n])

        #print('a-a*:')
        #print(a_minus_a_star)
        #print('a:')
        #print(a)
        # selecting support vectors

        # numeric error of the solution
        numeric_error = self.C * 0.00001
        self.index = np.linspace(0,n-1,n)

        # conditions of support vectors
        msk = np.squeeze(((self.a>numeric_error) & (self.a<self.C-numeric_error)) | ((self.a_star>numeric_error) & (self.a_star<self.C-numeric_error)))
        self.support_vectors = X[msk,:]
        w = self.a_minus_a_star[msk,:];
        self.support_vectors_index = self.index[msk]

        msk = np.squeeze(((self.a>numeric_error) & (self.a<self.C-numeric_error)))
        self.a_support_vectors = X[msk,:];
        self.a_support_vectors_index = self.index[msk]

        msk = np.squeeze(((self.a_star>numeric_error) & (self.a_star<self.C-numeric_error)))
        self.a_star_support_vectors = X[msk,:];
        self.a_star_support_vectors_index = self.index[msk]

        # calculating parameter b - KKT conditions
        if(np.size(self.a_support_vectors) != 0):
            fx = 0
            for i in self.index:
                i = int(i)
                fx = fx + self.a_minus_a_star[i]*K(X[i,:],self.a_support_vectors[0,:],self.kernel,self.degree,self.gamma)
            self.b = float(y[int(self.a_support_vectors_index[0])]-fx-self.epsilon)

        else:

            fx = 0
            for i in self.index:
                i = int(i)
                fx = fx + self.a_minus_a_star[i]*K(X[i,:],self.a_star_support_vectors[0,:],self.kernel,self.degree,self.gamma)
            self.b = -float(-y[int(self.a_star_support_vectors_index[0])]+fx-self.epsilon)

    def predict(self,X):

        # plotting test trajectory
        result = np.zeros(np.size(X,0))
        j = 0
        for x in X:
            fx = 0
            for i in self.index:
                i = int(i)
                dot = K(self.X[i,:],x,self.kernel,self.degree,self.gamma)
                fx = fx + self.a_minus_a_star[i] * dot
            result[j] = fx+self.b
            j += 1

        return result

    def get_support_vectors(self):
        return self.support_vectors

    def get_support_vectors_index(self):
        return self.support_vectors_index
