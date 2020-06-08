# Algorithms comparison with stratified cross-validation

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# custom SVR
from custom_SVR import SVR
# scikit-learn SVR
from sklearn.svm import SVR as skSVR

# reading data
wines_data = pd.read_csv('winequality/winequality-red.csv', sep=';')
full_set = wines_data.values
X = full_set[:,:-1]
y = full_set[:,-1]
classes = np.unique(full_set[:,-1])

# general params of regressor
eps = 0.5;
C = 10

# kernels to compare
kernels = ['linear', 'poly', 'rbf']

# polynomial maximum degree
degs = 3

# Stratified cross-validation
# number of folds
k = 3
skf = StratifiedKFold(n_splits = k)

res_str = "Results: \n"
labels = []
errors = np.zeros((2, degs+len(kernels)-1))
custom_me_for_classes = np.zeros((degs+len(kernels)-1, len(classes), k))
me_for_classes = np.zeros((degs+len(kernels)-1, len(classes), k))

l = 0
for kern in kernels:

  customerror = np.zeros(k)
  error = np.zeros(k)
  i = 0

  if (kern == 'poly'): # polynomial kernels
    for j in range(1, degs+1):
      deg = j
      i = 0
      # Cross-validation
      for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # creating regressors
        customregressor = SVR(epsilon = eps, kernel = kern, degree = deg, C=C)
        regressor = skSVR(epsilon = eps, kernel = kern, degree = deg, C=C)

        # learning process (may take a while)
        customregressor.fit(X_train,y_train)
        regressor.fit(X_train,y_train)

        # predicting
        customresult = customregressor.predict(X_test)
        result = regressor.predict(X_test)

        # errors
        customerror[i] = np.mean(np.absolute(customresult-y_test))
        error[i] = np.mean(np.absolute(result-y_test))

        for c, cls in enumerate(classes):
          pos = [p for p,v in enumerate(y_test) if v==cls]
          custom_me_for_classes[l, c, i] = np.mean(np.absolute(customresult[pos] - y_test[pos]))

        for c, cls in enumerate(classes):
          pos = [p for p,v in enumerate(y_test) if v==cls]
          me_for_classes[l, c, i] = np.mean(np.absolute(result[pos] - y_test[pos]))
        
        i += 1
      
      kernel_name = kern + "-" + str(j)
      res_str += "Mean error for custom model with " + kernel_name + " kernel: " + np.array2string(np.mean(customerror)) + "\n"
      res_str += "Mean error for model with " + kernel_name + " kernel: " + np.array2string(np.mean(error)) + "\n"
      labels.append(kernel_name)
      errors[0, l] = np.mean(customerror)
      errors[1, l] = np.mean(error)
      l += 1
  else: # other than polynomial kernels
    i = 0
    # Cross-validation
    for train_index, test_index in skf.split(X, y):c
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # creating regressors
      customregressor = SVR(epsilon = eps, kernel = kern, C=C)
      regressor = skSVR(epsilon = eps, kernel = kern, C=C)

      # learning process (may take a while)
      customregressor.fit(X_train,y_train)
      regressor.fit(X_train,y_train)

      # predicting
      customresult = customregressor.predict(X_test)
      result = regressor.predict(X_test)

      # error
      customerror[i] = np.mean(np.absolute(customresult-y_test))
      error[i] = np.mean(np.absolute(result-y_test))

      for c, cls in enumerate(classes):
          pos = [p for p,v in enumerate(y_test) if v==cls]
          custom_me_for_classes[l, c, i] = np.mean(np.absolute(customresult[pos] - y_test[pos]))

      for c, cls in enumerate(classes):
          pos = [p for p,v in enumerate(y_test) if v==cls]
          me_for_classes[l, c, i] = np.mean(np.absolute(result[pos] - y_test[pos]))

      i += 1

    res_str += "Mean error for custom model with " + kern + " kernel: " + np.array2string(np.mean(customerror)) + "\n"
    res_str += "Mean error for model with " + kern + " kernel: " + np.array2string(np.mean(error)) + "\n"
    labels.append(kern)
    errors[0, l] = np.mean(customerror)
    errors[1, l] = np.mean(error)
    l += 1

print(res_str)

# Ploting results for individual classes

kernels = ['linear', 'poly-1', 'poly-2', 'poly-3', 'rbf']
me_for_classes_mean = np.zeros((5, 6))
for i in range(0, 5):
  for j in range(0, 6):
    me_for_classes_mean[i, j] = np.mean(me_for_classes[i,j])

custom_me_for_classes_mean = np.zeros((5, 6))
for i in range(0, 5):
  for j in range(0, 6):
    custom_me_for_classes_mean[i, j] = np.mean(custom_me_for_classes[i,j])

for i in range(0, 5):

  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])

  x_axis = classes
  plt.xticks(classes, ['class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8'])
  ax.bar(x_axis - 0.1, custom_me_for_classes_mean[i,:], color = 'b', width = 0.2)
  ax.bar(x_axis + 0.1, me_for_classes_mean[i,:], color = 'g', width = 0.2)
  ax.legend(['Custom SVR algorithm', 'Scikit-learn SVR algorithm'])
  plt.title('Algorithms mean differential error comparison for different classes, kernel ' + kernels[i], fontdict=None, loc='center')

plt.show()

# Ploting results for different kernels

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

ticks_labels = labels
x_axis = np.arange(1, len(ticks_labels)+1, step=1)
plt.xticks(x_axis, ticks_labels)
ax.bar(x_axis - 0.1, errors[0,:], color = 'b', width = 0.2)
ax.bar(x_axis + 0.1, errors[1,:], color = 'g', width = 0.2)
ax.legend(['Custom SVR algorithm', 'Scikit-learn SVR algorithm'])
plt.title('Algorithms mean differential error comparison', fontdict=None, loc='center')
plt.show()