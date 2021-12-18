"""
Implementation of KNN algorithm for regression problem.

Refer to https://realpython.com/knn-python/#a-step-by-step-knn-from-scratch-in-python
for more details.
"""

import pandas as pd
import numpy as np
from math import sqrt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import axis0_safe_slice
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor

def get_splt_data(df, label_col, remove_col=[]):
    df = df.drop(remove_col, axis=1)
    X = df.drop(label_col, axis=1).copy()
    y = df[label_col]

    return train_test_split(X, y, random_state=42)

def knn_fit_numpy(data, k=3):
    X_train = data[0]
    X_test = data[1]

    y_train = data[2]
    y_test = data[3]

    y_preds = []
    for i in X_test.values:
        # calculate distance for one test value with all train data points
        num_distances = np.linalg.norm(X_train - i, axis=1)

        # get the data point indexes of nearest 'k' values
        nearest_k = num_distances.argsort()[:k]

        # get the corresponding the training labels for the k-nearest
        k_nearest_neigbors = y_train.iloc[nearest_k]

        predicted_val_i = k_nearest_neigbors.mean()
        y_preds.append(predicted_val_i)

    mse = mean_squared_error(y_preds, y_test.values)
    rmse = sqrt(mse)
    print('RMSE with numpy [k = %d] - %.2f \n' % (k, rmse))

def knn_fit_sklearn(data, k=3):
    X_train = data[0]
    X_test = data[1]

    y_train = data[2]
    y_test = data[3]

    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_preds = knn.predict(X_test)

    mse = mean_squared_error(y_preds, y_test.values)
    rmse = sqrt(mse)
    print('RMSE with sklearn [k = %d] - %.2f \n' % (k, rmse))

def knn_fit_sklearn_cv(data):
    X_train = data[0]
    X_test = data[1]

    y_train = data[2]
    y_test = data[3]

    parameters = {"n_neighbors": range(1, 50)}
    gridcv = GridSearchCV(KNeighborsRegressor(), parameters)

    gridcv.fit(X_train, y_train)

    k = gridcv.best_params_['n_neighbors']
    print('Optimal K value from cross validation : [k = %d] \n' % k )

    y_preds = gridcv.predict(X_test)
    mse = mean_squared_error(y_preds, y_test.values)
    rmse = sqrt(mse)
    print('RMSE with cross validation [k = %d] - %.2f \n' % (k, rmse))
    return k

def knn_fit_with_bagging(data, best_k):
    # using KNN as estimator for bagging instead of DT

    X_train = data[0]
    X_test = data[1]

    y_train = data[2]
    y_test = data[3]

    knn = KNeighborsRegressor(n_neighbors=best_k)

    bagging_model = BaggingRegressor(knn, n_estimators=100)
    bagging_model.fit(X_train, y_train)
    y_preds = bagging_model.predict(X_test)
    mse = mean_squared_error(y_preds, y_test)
    rmse = sqrt(mse)
    print('RMSE with bagging [k = %d] - %.2f \n' % (best_k, rmse))

if __name__ == '__main__':
    df = pd.read_csv('data/abalone.data')
    x = get_splt_data(df, 'Rings', ['Sex'])
    
    knn_fit_numpy(x)
    knn_fit_sklearn(x)
    knn_fit_sklearn(x, k=10)
    best_k = knn_fit_sklearn_cv(x)
    knn_fit_with_bagging(x, best_k)