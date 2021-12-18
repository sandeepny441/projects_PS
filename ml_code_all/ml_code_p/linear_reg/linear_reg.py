"""
Variaous methods of modelling linear regerssion
algorithm.
Least Squares
Gradient descent
Using sklean LinearRegression class.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.linear_model

class LinearRegression():
    def __init__(self):
        pass

    """
    loads the given input csv 
    """
    def load_data(self, input_file = "", feature_col = [], label_col = ""):
        data = pd.read_csv(input_file)

        if len(feature_col) == 1:
            x = data[feature_col[0]]
        else:
            x = data[feature_col]
        y = data[label_col]
        print(x.shape, type(x))
        print(y.shape, type(y))
        return x, y

    """
    Implemented using the ordinary least squares equation.
    x - can be only one dimensional.
    """
    def least_squares_fit(self, x, y):
        x = x.to_numpy()
        y = y.to_numpy()
        
        n = len(x)
        sum_xy = sum(x * y)
        num = ((n * sum_xy) - (sum(x) * sum(y)))
        dem = (n * sum(x * x)) - (sum(x) * sum(x))

        m = (num / dem)
        c = (sum(y) - (m * sum(x))) / n
    
        print("R-square score - %d" % 
                self.measure_accuracy(x, y, m, c))
        return m ,c

    def gradient_descent_fit(self, x, y):
        # initial update
        m = 0
        c = 0

        # set learning rate
        alpha = 0.0001
        epochs = 1000

        n = float(len(x))
        for i in range(epochs):
            # calculate new prediction
            y_pred = (m * x) + c

            # get new update coefficients value
            D_m = ((-2/n) * sum(x * (y - y_pred)))
            D_c = ((-2/n) * sum(y - y_pred))

            # update parameters
            m = m - (alpha * D_m)
            c = c - (alpha * D_c)
        
        return m, c

    def scikit_learn_fit(self, x, y):
        split_res = train_test_split(x, y, test_size=0.25, random_state=4)
        X_train, y_train = split_res[0], split_res[2]

        # sklearn expect 2-D input
        X_train = X_train.to_numpy().reshape(-1, 1)

        print(X_train.shape, y_train.shape)
        reg = sklearn.linear_model.LinearRegression()
        reg.fit(X_train, y_train)
        m = reg.coef_[0]
        c = reg.intercept_

        x_test, y_test = split_res[1], split_res[3]
        x_test = x_test.to_numpy().reshape(-1, 1)
        print("R-suqre score - %d" % self.measure_accuracy(x_test, y_test, sklearn_model=reg))
        return m, c

    def measure_accuracy(self, x, y, m=0, c=0, sklearn_model=None):
        if sklearn_model:
            res = sklearn_model.score(x, y)
            return res * 100 # r-squared value
        else:
            ss_tot = 0
            ss_res = 0
            n = len(x)
            for i in range(n):
                y_pred = c + m * x[i]
                mean_y = np.mean(y)
                ss_tot += (y[i] - mean_y) ** 2
                ss_res += (y[i] - y_pred) ** 2
            r2 = 1 - (ss_res/ss_tot)
            return r2

    """
    Plots a line fit
    """
    def plot_x_y(self, X, Y, m , c, title_text, count):
        Y_pred = (m * X) + c
        plt.figure(count)
        # plotting the inputs and labels
        plt.scatter(X, Y, color='orange', label='Scatter Plot')

        x_range = [min(X), max(X)]
        y_range = [min(Y_pred), max(Y_pred)]

        # plotting the fitted line
        plt.plot(x_range, y_range, color='red', label='Regression Line')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title(title_text)
        

if __name__ == '__main__':
    reg = LinearRegression()
    x, y = reg.load_data('data/data.csv', ['X'], 'Y')
    m, c = reg.least_squares_fit(x, y)
    print('OLS - coeffiecient - ', m, c)
    reg.plot_x_y(x, y, m, c, "using least squares", 1)

    m, c = reg.gradient_descent_fit(x, y)
    print('gradient descent - coeffiecient - ', m, c)
    reg.plot_x_y(x, y, m, c, "using gradient descent", 2)

    m, c = reg.scikit_learn_fit(x, y)
    print('with scikit learn - coeffiecient - ', m, c)
    reg.plot_x_y(x, y, m, c, "using scikit learn", 3)
    
    # show all plots at end
    plt.show()
