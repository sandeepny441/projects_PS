"""
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import amax
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

class LogisticReg:
    def __init__(self, input_file, output_label, ignore_first_col=False):
        data = pd.read_csv(input_file)
        # get all data except the output label.
        print(data.shape)
        if ignore_first_col:
            # on some cases, data has serial number on first column.
            drop_colulms = [data.columns[0], output_label]
            X = data.drop(drop_colulms, axis=1)
        else:
            X = data.loc[:, data.columns != output_label]
        y = data[output_label]
        print("No of columns - %d" % X.shape[1])
        print("No of classes - %d" % len(pd.unique(y)))
        self.X = X
        self.y = y

    def model_with_linear(self):
        model = LogisticRegression(
                    solver = 'liblinear',
                    random_state=0)
        
        model.fit(self.X_train, self.y_train)
        
        print("Linear model")
        print("Train size : %s, %d" % (self.X_train.shape, len(self.y_train)))
        print("******* Model parameters *********")
        print(model.coef_)
        print(model.intercept_)
        print('Total parameters : %d' % 
                    (len(model.coef_[0]) + len(model.intercept_)))

        print("******* Accuracy measure *********")
        print(classification_report(self.y_test, model.predict(self.X_test)))
        self.plot_heat_map_binary(
           model, reg.X_test, reg.y_test, "linear model")
        #self.plot_roc_auc([model], reg.X_test, reg.y_test, "orange", ["linear model"])
        print("-" * 60)
        return model

    def model_with_newton_method(self):
        model = LogisticRegression(
                    solver = 'newton-cg',
                    random_state=0)
        model.fit(self.X_train, self.y_train)
        print("Newton method model parameters : ")
        print(model.coef_)
        print(model.intercept_)
        print('Total parameters : %d' % 
                    (len(model.coef_[0]) + len(model.intercept_)))

        print("******* Accuracy measure *********")
        self.plot_heat_map_binary(
                    model,
                    self.X_test,
                    self.y_test,
                    "newtons method")
        print(classification_report(self.y_test, model.predict(self.X_test)))
        #self.plot_roc_auc([model], reg.X_test, reg.y_test, "green", ["newton model"])
        print("-" * 60)
        return model

    def model_with_regularization(self):
        model = LogisticRegression(
                    solver = 'liblinear',
                    penalty="l2",
                    C=10.0, # 
                    random_state=0)
        model.fit(self.X_train, self.y_train)
        print("L2 regularization")
        print("**** model parameters ****")
        print(model.coef_)
        print(model.intercept_)
        print('Total parameters : %d' % 
                    (len(model.coef_[0]) + len(model.intercept_)))

        print("******* Accuracy measure *********")
        self.plot_heat_map_binary(
            model, self.X_test, self.y_test, "l2 - regularization")
        print(classification_report(self.y_test, model.predict(self.X_test)))
        print("-" * 60)
        return model

    def model_with_stats_model():
        pass

    def plot_heat_map_binary(self, model, X_test, y_test, title):
        cm = confusion_matrix(y_test, model.predict(X_test))
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm)
        ax.grid(False)
        ax.xaxis.set(
                ticks=(0, 1),
                ticklabels=('Predicted 0s', 'Predicted 1s'))

        ax.yaxis.set(
                ticks=(0, 1),
                ticklabels=('Actual 0s', 'Actual 1s'))

        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], 
                        ha='center', va='center', color='red')
        plt.title(title)

    def plot_roc_auc(self, models, X_test, y_test, colors, titles):
        #plt.style.use('seaborn')
        fig, ax = plt.subplots(1)
        for i, model in enumerate(models):
            predict_proba = model.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, predict_proba[:,1], pos_label=1)

            # for plotting the standard roc line (reference)
            random_probs = [0 for i in range(len(y_test))]
            p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)


        
            # plot roc curves
            ax.plot(fpr, tpr, linestyle='--',color=colors[i], label=titles[i])
            ax.plot(p_fpr, p_tpr, linestyle='--', color='blue')

            ax.set_title('ROC curve')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive rate')
            #ax.xlabel('False Positive Rate')
            #ax.ylabel('True Positive rate')

            ax.legend(loc='best')

if __name__ == '__main__':
    reg = LogisticReg(
            input_file='data/emails.csv',
            output_label='Prediction',
            ignore_first_col=True)

    split_data = train_test_split(reg.X, reg.y, test_size=0.3, random_state=0)

    reg.X_train = split_data[0]
    reg.X_test = split_data[1]
    reg.y_train = split_data[2]
    reg.y_test = split_data[3]

    models = []
    m1 = reg.model_with_linear()
    m2 = reg.model_with_newton_method()
    m3 = reg.model_with_regularization()

    from sklearn.neighbors import KNeighborsClassifier
    m4 = KNeighborsClassifier(n_neighbors=4)
    m4.fit(reg.X_train, reg.y_train)

    reg.plot_roc_auc(
        [m1, m2, m3, m4], 
        reg.X_test,
        reg.y_test,
        ['orange', 'green', 'yellow', 'red'],
        ['linear', 'newtons', 'regularization', 'KNN'])
    # show all plots at end
    plt.show()