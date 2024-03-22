import numpy as np
from fcmeans import FCM
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
class boostOBU():
    def __init__(self,
                 base_estimator=SVC(kernel="rbf",C = 1.0),
                 random_state=None,kind=1):
        self.base_estimator =base_estimator
        self.imbalance = [1.5,3,12,30,60,120,200,400]
        self.train_x_1 = []
        self.train_x_2 = []
        self.train_y_1 = []
        self.train_y_2 = []
        self.random_state = random_state
        self.kind = kind
    def fit(self,X, y, label_maj=0, label_min=1):
        try:
            X,y = BorderlineSMOTE(k_neighbors=3).fit_resample(X,y)
        except:
            X,y = BorderlineSMOTE(k_neighbors = len(y[y==label_min])-1).fit_resample(X,y)
        X_min = X[y==label_min]
        X_maj = X[y==label_maj]
        y_min = y[y==label_min]
        y_maj = y[y==label_maj]
        fcm = FCM(n_clusters=2)
        fcm.fit(X)
        _labels = fcm.u.argmax(axis=1)
        membership = fcm.u[y==label_maj]
        maj_labels = _labels[y==label_maj]
        mean1 = membership[:,0].mean()
        mean2 = membership[:,1].mean()
        threshold = min(mean1,mean2)
        # c1 = np.where(maj_labels==0)[-1].tolist()
        # c2 = np.where(maj_labels==1)[-1].tolist()
        cond1 = membership[:,0] >threshold
        cond2 = membership[:,1] >threshold

        select1 = X_maj[maj_labels == 0 | (maj_labels == 1 & cond1),]
        select2 = X_maj[maj_labels == 1 | (maj_labels == 0 & cond2),]
        new_y_maj1 = np.full(select1.shape[0], y_maj[0])
        new_y_maj2 = np.full(select2.shape[0], y_maj[0])
        self.train_x_1 = np.concatenate((select1 , X_min),axis=0)
        self.train_y_1 = np.concatenate((new_y_maj1 , y_min),axis=0)
        self.train_x_2 = np.concatenate((select1, X_min),axis=0)
        self.train_y_2 = np.concatenate((new_y_maj2, y_min),axis=0)

        self.base_estimator = DecisionTreeClassifier()

        if self.kind == 1:
            self.base_estimator.fit(self.train_x_1, self.train_y_1)
        elif self.kind == 2:
            self.base_estimator.fit(self.train_x_2, self.train_y_2)
        else:
            raise ValueError(
                f'The predict "kind" of algorithm are '
                f'"1" and "2".'
                f"Got {self.kind} instead."
            )
        return self
    # def predict_proba(self, X):
    #
    #     w = np.array(self.weight_)
    #     w = w / w.sum()
    #     #        print(f"w_len:{w},#estimators:{len(self.estimators_)}")
    #     y_pred = np.array(
    #         [model.predict_proba(X) * w[i] for i, model in enumerate(self.estimators_)]
    #     ).sum(axis=0)
    #     # y_pred = np.array(self.estimators_[-1].predict_proba(X))
    #     return y_pred
    def predict_proba(self, X , kind=1):
        # g=X.shape[1]
        # self.base_estimator=SVC(kernel="rbf",C = 1.0,gamma=g,probability=True)


        y_pred = self.base_estimator.predict_proba(X)
        return y_pred

    def predict(self, X):
        """Predict class for X.
        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.
        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        y_pred = self.predict_proba(X)
        return y_pred.argmax(axis=1)

