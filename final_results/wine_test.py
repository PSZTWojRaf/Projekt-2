# this script tests how the custom regressor performs on wine_set

import pandas as pd
import xlrd

import numpy as np
import math

import matplotlib.pyplot as plt
import os

from sklearn.svm import LinearSVR, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# reading data
wines_data = pd.read_csv('winequality/winequality-red.csv', sep=';')
full_set = wines_data.values
X = full_set[:,:-1]
y = full_set[:,-1]
classes = np.unique(full_set[:,-1])

# split into train and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X, y, random_state=0)

# custom SVR
from custom_SVR import SVR

# params of regressor
eps = 0.5;
kern = 'linear';
# kern = 'poly';
# kern = 'rbf';
deg = 2
C = 10

# creating regressor
regressor = SVR(epsilon = eps, kernel = kern, degree = deg, C=C)

# learning process (may take a while)
regressor.fit(X_train,y_train)

# predicting
result = regressor.predict(X_test)

# displaying the mean error
print(np.mean(np.absolute(result-y_test)))
