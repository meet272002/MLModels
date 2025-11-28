import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Created By Meet Gandhi on 28/11/2025 | Class Creation for linear regression code.
class LinearRegression:
    def __init__(self,learning_rate = .1,iteration = 1000):
        self.learning_rate = learning_rate
        self.iteration = iteration

    def fit(self,X,y):
        self.training_examples,self.features = X.shape[0],X.shape[1]

        self.W = np.zeros((self.features,1))
        self.b = 0
        self.X = X
        self.y = y
        self.m = X.shape[0]

        for i in range(self.iteration):
            self.update_weights()

        return self

    def update_weights(self):
        y_predict = self.predict(self.X)

        residual = y_predict - self.y

        gradient_w = (1/self.m)*(self.X.T)@residual
        gradient_b = np.sum(residual)/self.m

        self.W  = self.W - self.learning_rate*gradient_w
        self.b = self.b - self.learning_rate*gradient_b

        return self

    def predict(self,X):
        y_predicted = X @ self.W + self.b
        return y_predicted

def main():
    train = pd.read_csv("train.csv")
    X = train.drop(columns=['is_positive'])
    y = train[['is_positive']]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

    model = LinearRegression()
    model.fit(X,y)

    print(model.predict(X))

main()
