# -*- coding: utf-8 -*-
"""
@author: JinJunRen & JuTong
"""

from os import replace
import numpy as np
import scipy.sparse as sp
import sklearn
from collections import Counter
from tools.imbalancedmetrics import ImBinaryMetric
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings
import matplotlib.pyplot as plt


from opfython.models import SupervisedOPF
warnings.filterwarnings("ignore")


class TSSE(BaseEstimator, ClassifierMixin):
    """ TSSE-BIM
    Parameters
    ----------
    base_estimator : object, optional (default=sklearn.Tree.DecisionTreeClassifier())
        The base estimator to fit on self-paced under-sampled subsets of the dataset.
        NO need to support sample weighting.
        Built-in `fit()`, `predict()`, `predict_proba()` methods are required.
    n_estimators :  integer, optional (default=10)
        The number of base estimators in the ensemble.
    random_state :  integer / RandomState instance / None, optional (default=None)
        If integer, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by
        `numpy.random`.
    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.
    estimators_ : list of estimator
        The collection of fitted base estimators.
    Example:
    ```
    import numpy as np
    from sklearn import datasets
    from sklearn.tree import DecisionTreeClassifier
    from utils import make_binary_classification_target, imbalance_train_test_split
    X, y = datasets.fetch_covtype(return_X_y=True)
    y = make_binary_classification_target(y, 7, True)
    X_train, X_test, y_train, y_test = imbalance_train_test_split(
            X, y, test_size=0.2, random_state=42)
    tsse = TSSE(
        base_estimator=DecisionTreeClassifier(),
        n_estimators=10,
        random_state=42,
    ).fit(
        X=X_train,
        y=y_train,
    )
    print('auc_prc_score: {}'.format(tsse.score(X_test, y_test)))
    ```
    """
    base_estimator = DecisionTreeClassifier(max_depth=None)
    def __init__(self,
                 base_estimator=DecisionTreeClassifier(),
                 n_estimators=10,
                 random_state=None):
        self.base_estimator = base_estimator
        self.estimators_ = []
        self.weight_ = []
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.fp = np.array([])
        self.fn = np.array([])

    @classmethod
    def fit_base_estimator(self, X, y):
        """Private function used to train a single base estimator."""
        return sklearn.base.clone(self.base_estimator).fit(X, y)

    def random_under_sampling(self, X_maj, y_maj, X_min, y_min):
        """Private function used to perform random under-sampling."""
        maj_idx = self.random_sampling(len(X_maj), len(X_min))
        X_train = np.concatenate([X_maj[maj_idx], X_min])
        y_train = np.concatenate([y_maj[maj_idx], y_min])
        return X_train, y_train

    def random_sampling(self, X_len, n):
        if n>X_len:
            print("*****************************************************************************")
            idx = np.random.choice(X_len,n,replace=True)
        else:
            idx = np.random.choice(X_len, n, replace=False)
        return idx

    def get_samples(self, prob, sampling_cnt,flag,X_maj):
        '''
        Parameters
        ----------
        prob : {array-like} of shape = [majority_class_samples]
            The majority class samples is the probability of the majority class, which
            are produced by the current ensemble.
        sampling_cnt : int,
            The number of the samples needing sampling, which equal the number of the minortiy class.


        Returns
        ------
        sampled_bins : {array-like} of shape = [sampling_cnt], the indexs of the majority class after sampling
        '''
        sampled_bins = []
        step = (prob.max() - prob.min()) / sampling_cnt
        interval = np.arange(prob.min(), prob.max(), step)  # interval of bins
        part_bin = np.digitize(prob, interval)  # partition probability of the majority class
        part_cnt = Counter(part_bin)  # count the number of samples in each bin (e.g., {bin_id:cnt,...})

        # calculate s, which store the number of the sampling samples in each bins
        s = np.zeros(sampling_cnt + 2)
        noempty_bin_key = [i for i in part_cnt.keys()]
        noempty_bin_key.sort()
        s[noempty_bin_key] = 1
        s = s * (1.0 / (len(part_cnt))) * self.s_bins
        # print(len(noempty_bin_key))

        if flag ==0:
            wights = []
            ave_contributions = []
            for i in range(0,len(noempty_bin_key)):
                # print("******",len(prob[part_bin==noempty_bin_key[i]]))
                ave_contributions.append(np.mean(prob[part_bin==noempty_bin_key[i]]))
                if ave_contributions[i] < 0.2:
                    wights.append(0)
                elif ave_contributions[i] >= 0.2 and ave_contributions[i] < 0.4:
                    wights.append(0.5)
                elif ave_contributions[i] >= 0.4 and ave_contributions[i] < 0.6:
                    wights.append(1)
                elif ave_contributions[i] >= 0.6 and ave_contributions[i] < 0.7:
                    wights.append(5)
                elif ave_contributions[i] >= 0.7 and ave_contributions[i] < 0.95:
                    wights.append(5)
                elif ave_contributions[i] >= 0.95 :
                    wights.append(2)

            for i in range(0, len(noempty_bin_key)):
                s[noempty_bin_key[i]] *= wights[i]
            s = np.ceil(s).astype(int)
        if flag != 0:
            s = np.ceil(s).astype(int)+1
        # s=s.astype(int)
        # print(s[noempty_bin_key])
        start_s_index = 1  # the index of the first sampling count
        pre_key = noempty_bin_key[0]  # the previous index of the noempty bins

        for key in noempty_bin_key:
            temp_elements = []
            ele_cnt = part_cnt[key]
            cur_needs = s[start_s_index:key].sum()
            start_s_index = key + 1  # record the next bin which will be sampled.
            # The bins from pre_index to key-1 need to sample, but there aren't enough samples in them.
            if cur_needs > 0 :
                if pre_key == 0:  # pre_key is the last bin,need  to start from the first element in part_bin
                    temp_elements = np.where(part_bin == (len(noempty_bin_key) - 1))[-1].tolist()
                else:
                    temp_elements = np.where(part_bin == key)[-1].tolist()
                sampled_bins.append(np.random.choice(
                    temp_elements, cur_needs, replace=True))

            if s[key] <= ele_cnt:  # the number of the samples in currrent bin are greater than that of needing samples.

                sampled_bins.append(np.random.choice(
                    np.where(part_bin == key)[-1].tolist(),
                    s[key],
                    replace=False))
            else:
                temp_elements += np.where(part_bin == key)[-1].tolist()
                if flag == 3:
                    sampled_bins.append(np.random.choice(
                    temp_elements, ele_cnt, replace=False))
                else:
                    sampled_bins.append(np.random.choice(
                        temp_elements, s[key], replace=True))
            pre_key = key
        # remaining bins
        if start_s_index < sampling_cnt :
            cur_needs = s[start_s_index:sampling_cnt].sum()
            temp_elements = np.where(part_bin == pre_key)[-1].tolist()
            sampled_bins.append(np.random.choice(
                temp_elements, cur_needs, replace=True))

        return sampled_bins

    def equalization_sampling(self, X_maj, y_maj,flag):

        prob_maj = self.y_pred_maj
        # If the probabilitys are not distinguishable, perform random smapling
        if prob_maj.max() == prob_maj.min():
            maj_idx = self.random_sampling(len(X_maj), self.k_bins)
            new_X_maj = X_maj[maj_idx]
        else:
            maj_sampled_bins = self.get_samples(prob_maj, self.k_bins,flag,X_maj)
            index = np.concatenate(maj_sampled_bins, axis=0)
            new_X_maj = X_maj[index]
        new_y_maj = np.full(new_X_maj.shape[0], y_maj[0])

        return new_X_maj, new_y_maj

    def fit(self, X, y, label_maj=0, label_min=1):

        self.estimators_ = []
        self.weight_ = []
        self.hard = []
        # Initialize by spliting majority / minority set
        X_maj = X[y == label_maj]
        y_maj = y[y == label_maj]
        X_min = X[y == label_min]
        y_min = y[y == label_min]
        self.k_bins = X_min.shape[0]
        self.s_bins=X_min.shape[0]
        self.y_pred_maj = np.zeros(X_maj.shape[0])
        self.y_pred_min = np.zeros(X_min.shape[0])
        for i_estimator in range(0, self.n_estimators):
            if i_estimator == 0:
                X_train, y_train = self.random_under_sampling(X_maj, y_maj, X_min, y_min)
                # from imblearn.under_sampling import OneSidedSelection, InstanceHardnessThreshold
                # X_train , y_train = OneSidedSelection().fit_resample(X,y)
                # X_train = np.vstack([new_maj_X, X_min])
                # y_train = np.hstack([new_maj_y, y_min])

                clf = self.fit_base_estimator(X_train, y_train)
                self.y_pred_maj = clf.predict_proba(X_maj)[:, 0]
                y_pred = clf.predict(X)
                im_metric = ImBinaryMetric(y, y_pred)
                eps = im_metric.gmean()
                self.fp = np.append(self.fp, im_metric.FP)
                self.fn = np.append(self.fn, im_metric.FN)
                self.weight_.append(1)
                self.estimators_.append(clf)
                continue
            else:
                if i_estimator > self.n_estimators/2:
                    new_maj_X, new_maj_y = self.equalization_sampling(X_maj, y_maj,0)
                else:
                    new_maj_X, new_maj_y = self.equalization_sampling(X_maj, y_maj, 1)
                # ipdb.set_trace()
                print(len(new_maj_y), len(X_min))
                X_train = np.vstack([new_maj_X, X_min])
                y_train = np.hstack([new_maj_y, y_min])
                clf = self.fit_base_estimator(X_train, y_train)

                y_pred = clf.predict(X)
                im_metric = ImBinaryMetric(y, y_pred)
                eps = im_metric.gmean()
                self.fp = np.append(self.fp, im_metric.FP)
                self.fn = np.append(self.fn, im_metric.FN)
                # print(self.fp,self.fn)
                w_a = (1 - self.fp / np.max(self.fp))
                w_a[w_a != w_a] = 1
                w_b = (1 - self.fn / np.max(self.fn))
                w_b[w_b != w_b] = 1
                _w = w_a * w_b
                if _w.max() == 0:
                    _w[:] = 1
                self.weight_ = _w
                temp_y_pred_maj = clf.predict_proba(X_maj)[:, 0]
                # self.weight_.append(eps)
                W = np.array(self.weight_)
                self.estimators_.append(clf)
                self.y_pred_maj = self.y_pred_maj * (W[0:-1].sum() / W.sum()) + temp_y_pred_maj * (W[-1] / W.sum())
        print('wight:', self.weight_)
        return self

    def predict_proba(self, X):

        w = np.array(self.weight_)
        w = w / w.sum()
        y_pred = np.array(
            [model.predict_proba(X) * w[i] for i, model in enumerate(self.estimators_)]
        ).sum(axis=0)
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

    def score(self, X, y):
        """Returns the average precision score (equivalent to the area under
        the precision-recall curve) on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.
        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.
        Returns
        -------
        score : float
            Average precision of self.predict_proba(X)[:, 1] wrt. y.
        """
        return average_precision_score(
            y, self.predict_proba(X)[:, 1])
