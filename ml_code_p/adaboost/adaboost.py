"""
refer to this link for more info:
https://towardsdatascience.com/adaboost-from-scratch-37a936da3d50
"""
import numpy as np
from numpy.core.defchararray import index
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

class AdaBoost():
    def __init__(self):
        # set of classifiers (decision trees)
        self.G_M = []

        # list of weights to be used for each classifier.
        self.alphas = []

        # number of iterations of boosting
        self.M = None

        # error on training for each iteration
        self.training_errors = []

    def fit(self, X, y, M=100):
        self.alphas = []
        self.M = M
        self.G_M = []

        for m in range(M):
            if m== 0:
                # first time, initialize weights to default
                w_i = (np.ones(len(y)) * 1) / len(y)
            else:
                w_i = AdaBoost.update_weights(w_i, alpha_m, y, y_pred)

            #print('using weights - ', w_i.shape)
            # fit a basic DT classifer
            G_m = DecisionTreeClassifier(max_depth=1) # stump with only one level.
            G_m.fit(X, y, sample_weight=w_i)
            y_pred = G_m.predict(X)

            self.G_M.append(G_m)

            # error for this
            error_m = AdaBoost.compute_error(y, y_pred, w_i)
            #print('error rate at %d - %d' % (m, error_m))
            alpha_m = AdaBoost.compute_alpha(error_m)
            #print('alpha rate at %d - %d' % (m, alpha_m))
            self.alphas.append(alpha_m)

        if len(self.alphas) != len(self.G_M):
            print('error - something wrong in fit (not enough weights for boosting')

    def predict(self, X):
        """
        Compute the overall adaboost prediction based on the classifiers prediction and weights
        Refer to the link in description for formula.
        """
        weak_preds = pd.DataFrame(index=range(len(X)), columns = range(self.M))
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:, m] = y_pred_m  # update for this iteration

        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)
        return y_pred

    @staticmethod
    def compute_error(y, y_pred, w_i):
        """
        Computes the error for a given classifier results.
        For formula, refer to the link in description
        """

        return (sum(w_i * (np.not_equal(y, y_pred)).astype(int))) / sum(w_i)

    @staticmethod
    def compute_alpha(error):
        """
        Computes the alpha to used with each weak classifier.
        """
        return np.log((1 - error) / error)

    @staticmethod
    def update_weights(w_i, alpha, y, y_pred):
        """
        Computes the weights to be used for next iteration
        """
        temp = np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))
        return w_i * temp

if __name__ == '__main__':
    df = pd.read_csv('data/spambase.data', header=None)

    # read columns
    names = pd.read_csv('data/spambase.names', sep = ':', skiprows=range(0, 33), header=None)
    col_names = list(names[0])
    col_names.append('Spam') # predictor label

    df.columns = col_names

    # convert the zero columns value to -1 (not spam)
    df['Spam'] = df['Spam'] * 2 - 1

    X_train, X_test, y_train, y_test = train_test_split(
                                                    df.drop(columns = 'Spam').values, 
                                                    df['Spam'].values, 
                                                    train_size = 3065, 
                                                    random_state = 2)

    ab = AdaBoost()
    print('Input data : ')
    print('\t %d examples with %d features' %(X_train.shape[0], X_train.shape[1]))
    ab.fit(X_train, y_train, M = 400) # using 400 DT's for boosting

    print('Test data : ')
    print('\t %d examples with %d features' %(X_test.shape[0], X_test.shape[1]))
    y_pred = ab.predict(X_test)

    error_rate = sum((np.not_equal(y_test, y_pred)).astype(int)) / len(y_test)
    print('with numpy impl   : Error rate %.2f %%' % (round(error_rate * 100, 2)))

    # correct_preds = (pd.Series(y_test) == y_pred).value_counts()
    # accuracy = correct_preds.values[0] / len(y_test)
    # print(accuracy)
    # print('with numpy impl : Accuracy %.2f %%' % (accuracy) * 100)

    # by default uses DT as classifier.
    ab_sklearn = AdaBoostClassifier(n_estimators=400, random_state=1)
    ab_sklearn.fit(X_train, y_train)

    y_pred = ab_sklearn.predict(X_test)

    error_rate = sum((np.not_equal(y_test, y_pred)).astype(int)) / len(y_test)
    print('with sklearn impl : Error rate %.2f %%' % (round(error_rate * 100, 2)))
    print('with sklearn impl : Accuracy %.2f %%' % (ab_sklearn.score(X_test, y_test) * 100))
