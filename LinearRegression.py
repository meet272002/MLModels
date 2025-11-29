import pandas as pd
import numpy as np
from train_test_split import model_selection as ms
from sklearn.preprocessing import StandardScaler

# Created By Meet Gandhi on 28/11/2025 | Class Creation for linear regression code.
# Updated By Meet Gandhi on 29/11/2025 | Major changes and error Solving.
class LinearRegression:
    def __init__(self,learning_rate = .1,iteration = 1000):
        self.learning_rate = learning_rate
        self.iteration = iteration

    def fit(self,X,y):
        self.training_examples,self.features = X.shape[0],X.shape[1]

        self.W = np.zeros((self.features,1))
        self.b = 0.0
        self.X = X
        self.y = y.values.flatten()
        self.m = X.shape[0]

        for i in range(self.iteration):
            self.update_weights()

        return self

    def update_weights(self):
        y_predict = self.predict(self.X)

        residual = y_predict - self.y

        residual_col_vector = residual.reshape(-1, 1)
        gradient_w = (1/self.m)*(self.X.T)@residual_col_vector
        gradient_b = np.sum(residual_col_vector)/self.m

        self.W  = self.W - self.learning_rate*gradient_w
        self.b = self.b - self.learning_rate*gradient_b

        return self

    def predict(self,X):
        y_predicted = X @ self.W + self.b
        return y_predicted.values.flatten()

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

    model = LinearRegression(learning_rate=.00001,iteration=10)
    model.fit(X_train,Y_train)

    y_predicted = model.predict(X_test)

    error = (y_predicted - Y_test)**2

    final = sum(error)/Y_test.shape[0]
    print(final)
main()
