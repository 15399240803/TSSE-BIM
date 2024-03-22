import numpy as np
from fcmeans import FCM
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import BorderlineSMOTE,ADASYN
from sklearn.base import BaseEstimator, ClassifierMixin
class BS():
    def __init__(self,
                 base_estimator=SVC(kernel="rbf",C = 1.0),
                 random_state=42):
        self.base_estimator =base_estimator
        self.train_x = []
        self.train_y= []
        self.random_state = random_state

    def fit(self,X, y, label_maj=0, label_min=1):
        try:
            self.train_x ,self.train_y = BorderlineSMOTE(k_neighbors=5).fit_resample(X,y)
        except:
            self.train_x ,self.train_y = BorderlineSMOTE(k_neighbors = len(y[y==label_min])-1).fit_resample(X,y)

        self.base_estimator = DecisionTreeClassifier()

        self.base_estimator.fit(self.train_x, self.train_y)

        return self
    def predict_proba(self, X , kind=1):
        # g=X.shape[1]
        # self.base_estimator=SVC(kernel="rbf",C = 1.0,gamma=g,probability=True)

        y_pred = self.base_estimator.predict_proba(X)
        return y_pred

    def predict(self, X):

        y_pred = self.predict_proba(X)
        return y_pred.argmax(axis=1)
class AS():
    def __init__(self,base_estimator=SVC(kernel="rbf",C = 1.0),
                 random_state=42,):
        self.train_x = []
        self.train_y= []
        self.random_state = random_state

    def fit(self,X, y, label_maj=0, label_min=1):
        try:
            self.train_x ,self.train_y = ADASYN(n_neighbors=5).fit_resample(X,y)
        except:
            self.train_x ,self.train_y = ADASYN(n_neighbors = len(y[y==label_min])-1).fit_resample(X,y)

        self.base_estimator = DecisionTreeClassifier()

        self.base_estimator.fit(self.train_x, self.train_y)

        return self
    def predict_proba(self, X , kind=1):
        # g=X.shape[1]
        # self.base_estimator=SVC(kernel="rbf",C = 1.0,gamma=g,probability=True)

        y_pred = self.base_estimator.predict_proba(X)
        return y_pred

    def predict(self, X):

        y_pred = self.predict_proba(X)
        return y_pred.argmax(axis=1)
