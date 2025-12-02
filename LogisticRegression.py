import pandas as pd
import numpy as np
from train_test_split import model_selection as ms
from sklearn.preprocessing import StandardScaler

class LogisticRegressin:
    def __init__(self,learning_rate,iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        self.cost_history = []

    def cost(self,h,y):
        m = len(y)
        return -1/m*np.sum(y*np.log(h) + (1-y)*np.log(1-h))

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def fit(self,X,y):
        m,n = X.shape

        self.weights = np.zeros(n)

        for _ in range(self.iterations):
            z = X@self.weights + self.bias
            h = self.sigmoid(z)

            dw = 1/m*(X.T@(h-y))
            db = 1/m*np.sum(h-y)

            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db

            self.cost_history.append(self.cost(h,y))

    def predict(self,X):
        return (self.sigmoid((X@self.weights) + self.bias) >= .5).astype(int)


def main():
    train = pd.read_csv("train.csv")

    numeric_cols = []
    for col in train.columns:
        if train[col].dtype != 'object' and col != 'is_positive' and col != 'review_id':
            numeric_cols.append(col)

    train.dropna(inplace=True)
    X = train[numeric_cols]
    y = train['is_positive']

    ss = StandardScaler()
    X_scaled = pd.DataFrame(ss.fit_transform(X),columns=X.columns,index=X.index)

    X_train, X_test, Y_train, Y_test = ms.train_test_split(ms,X_scaled,y,test_size = .3)

    model = LogisticRegressin(learning_rate=.00001,iterations=10)
    model.fit(X_train,Y_train)

    y_predicted = model.predict(X_test)

    error = (y_predicted - Y_test)**2

    final = sum(error)/Y_test.shape[0]
    print(final)
main()
