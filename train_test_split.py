import pandas as pd
import numpy as np

# Created by Meet Gandhi on 29/11/2025 | Class for train_test_split for teh X and Y dataset.
class model_selection:
    def train_test_split(self,X,y,test_size):
        total_rows = X.shape[0]
        test_rows = round(total_rows*test_size)

        shuffled_indices = np.random.permutation(total_rows)

        # Splitting dataset for X
        test_indices = shuffled_indices[0:test_rows]
        train_indices = shuffled_indices[test_rows:]

        # print(f"{len(test_indices)} and Test Size: {test_rows}")
        #Splitting dataset for X
        X_test = X.iloc[test_indices]
        X_train = X.iloc[train_indices]

        # Splitting dataset for y
        y_test = y.iloc[test_indices]
        y_train = y.iloc[train_indices]

        return X_train,X_test,y_train,y_test

df = pd.read_csv("train.csv")
X = df.drop(columns=['is_positive'])
y = df['is_positive']

ms = model_selection()
X_train,X_test,y_train,y_test = ms.train_test_split(X,y,test_size=.3)
