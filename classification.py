import codecs

from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd
import joblib


class Classification:

    def __init__(self, algorithm, x, y, x_test, y_test):
        """
        Constructor
        :param algorithm:
        :param file: features file
        """
        self.algorithm = algorithm
        self.x_train = x
        self.y_train = y
        self.x_test = x_test
        self.y_test = y_test

    # @staticmethod
    # def get_features(file):
    #     """
    #     Gets features from file
    #     :param file: feature file
    #     :return: array of features
    #     """
    #     #x = []
    #     df = pd.read_csv(file)
    #     # for line in data.readlines()[1:]:
    #     #    features = []
    #     #    features.append(float(line.split(';')[1].strip()))
    #     #    features.append(float(line.split(';')[2].strip()))
    #     #    x.append(features)
    #     x = df[['util', 'nao_util']]
    #     return x

    # @staticmethod
    # def get_label(file):
    #     """
    #     Gets labels from features file
    #     :param file: features file
    #     :return: array of labels
    #     """
    #     #y = []
    #     df = pd.read_csv(file)
    #     # for line in data.readlines()[1:]:
    #     #    y.append(float(line.split(';')[3].strip()))
    #     y = df[['classe']]
    #     return y

    @staticmethod
    def normalize(X):
        X_normalized = preprocessing.normalize(X, norm='l2', axis=0)
        return X_normalized
        # return X

    def classifier(self, save_folder_model=''):
        """
        Runs several classifiers (svm, naive bayes, decision tree, and nn)
        :param test_percentage: percentage of the test set
        :return: confusion matrix and cross validation k=10
        """
        # x_train, x_test, y_train, y_test = train_test_split(
        # self.x, self.y, test_size=test_percentage)
        print(len(self.x_train), len(self.x_test),
              len(self.y_train), len(self.y_test))
        if self.algorithm == 'svm':
            clf = svm.SVC(kernel='linear', gamma='auto')
            clf.fit(self.x_train, self.y_train)
            if save_folder_model != '':
                joblib.dump(clf, save_folder_model+'.model')
            print('Training: ', clf.score(self.x_test, self.y_test))
            y_pred = clf.predict(self.x_test)
        elif self.algorithm == 'naive_bayes':
            clf = GaussianNB()
            clf.fit(self.x_train, self.y_train)
            if save_folder_model != '':
                joblib.dump(clf, save_folder_model+'.model')
            print('Training: ', clf.score(self.x_test, self.y_test))
            y_pred = clf.predict(self.x_test)
        elif self.algorithm == 'tree':
            clf = DecisionTreeClassifier()
            clf.fit(self.x_train, self.y_train)
            if save_folder_model != '':
                joblib.dump(clf, save_folder_model+'.model')
            print('Training: ', clf.score(self.x_test, self.y_test))
            y_pred = clf.predict(self.x_test)
        elif self.algorithm == 'nn':
            clf = MLPClassifier(solver='adam', hidden_layer_sizes=(
                20, 20), random_state=42, max_iter=1000)
            clf.fit(self.x_train, self.y_train)
            if save_folder_model != '':
                joblib.dump(clf, save_folder_model+'.model')
            print('Training: ', clf.score(self.x_test, self.y_test))
            y_pred = clf.predict(self.x_test)
        elif self.algorithm == 'randfor':
            criterion = 'entropy'  # best fit
            estimators = 200  # best fit
            max_depth = None  # best fit
            clf = RandomForestClassifier(
                criterion=criterion, n_estimators=estimators, max_depth=max_depth)
            clf.fit(self.x_train, self.y_train)
            if save_folder_model != '':
                joblib.dump(clf, save_folder_model+'.model')
            print('Training: ', clf.score(self.x_test, self.y_test))
            y_pred = clf.predict(self.x_test)
        # cross_score = self.eval_cross_validation(clf)
        holdout_score = self.evaluation(y_pred, self.y_test)
        # cross_score,
        return (holdout_score)

    def eval_cross_validation(self, clf):
        """
        Runs cross validation k = 10
        :param clf: classifier
        :return: accuracy of classifier
        """
        scores = cross_val_score(clf, self.x, self.y, cv=10)
        return scores

    @staticmethod
    def evaluation(y_pred, y_test):
        """
        Results of classifiers
        :param y_pred: predicted labels
        :param y_test: test set
        :return: confusion matrix
        """
        #print(confusion_matrix(y_test, y_pred).ravel())
        return classification_report(y_test, y_pred, output_dict=True)
