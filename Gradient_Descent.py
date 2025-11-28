import numpy as np
import pandas as pd

# Created by Meet Gandhi on 28/11/2025 | Gradient Descent Method implementation.
def gradient_descent(X,y,learning_rate=0.1,n_iterations=1000):
    m = len(y)

    theta = np.zeros((X.shape[1],1))

    cost_history = []

    for i in range(n_iterations):
        prediction = X @ theta

        error = prediction - y

        gradient = (1/m)*X.T@error

        theta = theta - learning_rate*gradient

        cost = (1/2*m)*np.sum(error**2)
        cost_history.append(cost)

    return theta, cost_history

# Reshaping the array into 1 columns
X_raw = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([2, 4, 5, 4, 5]).reshape(-1, 1)

X_b = np.c_[np.ones((len(X_raw), 1)), X_raw]

learning_rate = 0.01
n_iterations = 1000
final_theta, history = gradient_descent(X_b, y, learning_rate, n_iterations)
print(f"Final parameters (Intercept, Slope): \n{final_theta.flatten()}")
print(f"Final Cost: {history[-1]:.4f}")