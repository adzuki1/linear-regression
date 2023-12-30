import numpy as np
import matplotlib.pyplot as plt

# Generate some fake data to fit the model
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# Add a column of ones to the inputs to represent the bias term
X = np.c_[np.ones((100, 1)), x]

# Compute the weights using the normal equation
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Predict the output for the inputs
y_pred = X.dot(theta)

# Plot the data and the model
plt.scatter(x, y, color='b')
plt.plot(x, y_pred, color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.show()