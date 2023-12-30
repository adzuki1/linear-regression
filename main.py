import numpy as np
import matplotlib.pyplot as plt


def generateData(seed=0, num_points=100):
    """Generate some fake data to fit the model"""

    np.random.seed(seed)
    x = np.random.rand(num_points, 1)
    y = 2 + 3 * x + np.random.rand(num_points, 1)

    return x, y


def plotResults(x, y, y_pred):
    """Plot the data and the model"""

    plt.scatter(x, y, color='b', label='Original Data')
    plt.plot(x, y_pred, color='k', label='Linear Regression Model')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def main():
    x, y = generateData()

    # Add a column of ones to the inputs to represent the bias term
    X = np.c_[np.ones((100, 1)), x]

    # Compute the weights using the normal equation
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    # Predict the output for the inputs
    y_predict = X.dot(theta)

    plotResults(x, y, y_predict)


if __name__ == "__main__":
    main()
