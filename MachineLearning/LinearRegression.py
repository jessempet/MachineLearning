import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros

def warmup():
    return np.eye(5)

def plot(x, y):
    plt.axis([5, 25, -5, 25])
    return plt.scatter(x,y)

def computeCost(X, y, theta, m):
    h = np.transpose(X)*theta
    return (1 / (2*m))*np.sum((h-y)**2);

def gradDescent(X, y, theta, alpha, num_iters):
   m = y.size
   J_history = zeros(shape=(num_iters, 1))
   for i in range(num_iters):
       predictions = X.dot(theta).flatten()
       errors_x1 = (predictions - y) * X[:, 0]
       errors_x2 = (predictions - y) * X[:, 1]
       theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
       theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()
       J_history[i, 0] = computeCost(X, y, theta, m)
   return theta, J_history

if __name__=='__main__':
    print("Running warmup Excercise ... \n")
    print("5x5 Identity Matrix: \n")
    print(warmup(), "\n")
    data = np.loadtxt("E:\Machine Learning\machine-learning-ex1\machine-learning-ex1\ex1\ex1data1.txt",delimiter=',')
    x = np.array(data[:,0])
    y = np.array(data[:,1])
    m = len(data)
    plot(x,y)
    plt.grid(b=None,which='both')

    X = np.transpose(np.vstack([np.ones(m),x]))
    theta = np.transpose([np.zeros(2)])
    print(theta)
    iterations = 1500
    alpha = 0.01

    print(computeCost(X, y, theta, m))
    theta = gradDescent(X, y, theta, alpha, iterations)

    print("Theta found by gradient descent:",str(theta[0][0]), str(theta[0][1]))

    predict1 = np.dot(np.array([1, 3.5]),theta[0]) * 10000
    print("For population = 35000, we predict a profit of ",str(predict1))

    plt.plot(X, np.dot(X,theta[0]))
    plt.show()




