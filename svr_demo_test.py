# this script shows how the custom regressor performs on 2-D test set

from custom_SVR import SVR
import numpy as np
import matplotlib.pyplot as plt

# preparing the data
X = np.linspace(-5,5,11)
y = X+np.random.normal(0,1,np.size(X))
X = np.concatenate(([X],[X]),axis = 0)
X = X.transpose()

# params of regressor
eps = 0.5;
# kern = 'linear';
kern = 'poly';
# kern = 'rbf';
deg = 5
C = 10
gam = 0.1

# test data
test_X = np.linspace(-5,5,51)
test_X = np.concatenate(([test_X],[test_X]),axis = 0)
test_X = test_X.transpose()
# creating regressor
regressor = SVR(epsilon = eps, kernel = kern, degree = deg, gamma=gam, C=C)

# learning process (may take a while)
regressor.fit(X,y)
# predicting
result = regressor.predict(test_X)
SV = regressor.get_support_vectors()
SV_idx = regressor.get_support_vectors_index()

# displaying the result


# original samples
plt.scatter(X[:,1], y, cmap='winter')
# predicted line
plt.plot(test_X[:,1], result, color='red')
# tube
plt.plot(test_X[:,1], result+eps, linestyle='dashed', color='red', linewidth=1)
plt.plot(test_X[:,1], result-eps, linestyle='dashed', color='red', linewidth=1)
# support vectors
plt.scatter(SV[:,1],y[SV_idx.astype(int)], color='green', linewidths=4)
plt.show()
