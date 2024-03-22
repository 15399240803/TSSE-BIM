# -*- coding: utf-8 -*-
"""
@author: JinJunRen & JuTong
"""
from sklearn.model_selection import KFold
import sys
import os
import argparse
import time
sys.path.append("..")
import pandas as pd
import tools.dataprocess as dp
from ensemble.self_paced_ensemble import SelfPacedEnsemble as SPE
from ensemble.equalizationensemble import EASE
from ensemble.tsse import TSSE
# from ensemble.ECUBoost_RF import ECUBoostRF
# from ensemble.boost_OBU import boostOBU
# from ensemble.rbu import RBU
# from ensemble.hub_ensemble import HashBasedUndersamplingEnsemble as HUE
from ensemble.canonical_ensemble import *
from ensemble.BS_AS import *
import numpy as np
from tools.imbalancedmetrics import ImBinaryMetric
from tqdm import tqdm
import warnings
from imblearn.under_sampling import OneSidedSelection,CondensedNearestNeighbour
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from imblearn.metrics import geometric_mean_score

# from imblearn.ensemble import
METHODS = ['EASE','TSSE', 'SMOTEBoost' , 'SMOTEBagging', 'RUSBoost', 'UnderBagging', 'BalanceCascade','AS','BS']
RANDOM_STATE = None

# from sklearnex import patch_sklearn
# patch_sklearn()
def parse():
    '''Parse system arguments.'''
    parse=argparse.ArgumentParser(
        description='General excuting Ensemble method', 
        usage='genealrunEnsemble.py -dir <datasetpath> -alg <algorithm name> -est <number of estimators>'
        ' -n <n-fold>'
        )
    #r'F:\PycharmProjects\ml\datasets\keel_dataset\two_class2\yeast-1_vs_7.csv'  page-blocks1.csv
    # r'F:\PycharmProjects\ml\EASE\creditcard_2.csv'   r'F:\PycharmProjects\ml\EASE\diabetes.csv'
    #F:\PycharmProjects\ml\syn_dataset\1\1+0.csv
    # bupa(liver).csv  normal_vs_r2l.csv  dos_vs_r2l.csv Record_Linkage.csv   churn.csv  diabetes.csv  Give_Me_Some_Credit!
    #F:\PycharmProjects\ml\syn_dataset  , mnist3.csv
    parse.add_argument("--dir",dest="dataset",default=r'F:\PycharmProjects\ml\datasets\keel_dataset\two_class2\ecoli2.csv',help="path of the datasets or a dataset")
    parse.add_argument("--alg",dest="algorithm",nargs='+',default=['TSSE'],help="list of the algorithm names ")
    parse.add_argument("--est",dest="estimators",default=10, type=int, help="number of estimators")
    parse.add_argument("--n",dest="nfold",default=5, type=int, help="n fold")
    return parse.parse_args()


def init_model(algname, params):

    '''return a model specified by "method".'''
    if algname in ['SPE']:
        model = eval(algname)(base_estimator =params['base_estimator'], 
                     k_bins=params['k_bins'], 
                     n_estimators =params['n_estimators'])
    elif algname in ['GB','RF']:
        model = eval(algname)(n_estimators =params['n_estimators'])
    elif algname in METHODS:
        model = eval(algname)()#,random_state=RANDOM_STATE
    elif algname =="ECUBoostRF":
        model = eval(algname)(L=params['n_estimators'],lamda=0.2,k=5,T=50)
    elif algname =="HUE":
        model = eval(algname)(base_estimator= params['base_estimator'],n_iterations=params['n_estimators'])
    elif algname == "RBU":
        model = eval(algname)()
    elif algname == "boostOBU":
        model = eval(algname)(kind = 1)
    else:
        print(f'No such method support: {algname}')
    return model


def main():
    para = parse()
    algs=para.algorithm
    datasetname=para.dataset
    pd.set_option('display.width', None)  # Setting the character display to be unlimited
    pd.set_option('display.max_rows', None)  # Setting the number of lines to display unlimited

    for alg in algs:
        # aaaa = pd.DataFrame(columns=['F1', 'std1', 'MMC', 'std2', 'AUC', 'std3', 'G_MEAN', 'std4'])
        aaaa = pd.DataFrame(columns=['F1', 'MMC', 'AUC', 'G_MEAN'])
        ds_path,ds_name=os.path.split(datasetname)
        print(alg)
        dataset_list=[]
        if os.path.isdir(datasetname):#is a set of data sets
            dataset_list=os.listdir(datasetname)
        else:#single data set
            dataset_list.append(ds_name)
        # pd.DataFrame(dataset_list).to_csv("dd.csv")
        for dataset in tqdm(dataset_list):
            traintimes=[]
            testtime = []
            scores = []
            metric_conf_m = []
            print(dataset)
            X,y=dp.readDateSet(ds_path+'/'+dataset)
            sss = StratifiedShuffleSplit(n_splits=para.nfold, test_size=0.2,random_state=RANDOM_STATE)
            fold_cnt=0

            for i in range(0,1):#Repeat n times
                scores1 = []

                for train_index, test_index in sss.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    fold_cnt+=1
                    params={'n_estimators':para.estimators, 'k_bins':10 ,'base_estimator':DecisionTreeClassifier()}
                    model = init_model(algname=alg,params=params)
                    start_traintime = time.perf_counter()
                    model.fit(X_train, y_train,)
                    traintimes.append(time.perf_counter()-start_traintime)
                    start_testtime = time.perf_counter()
                    y_pre=model.predict(X_test)
                    testtime.append(time.perf_counter()-start_testtime)
                    y_pred = model.predict_proba(X_test)[:, 1]#0 indicates the majority class，1 indicates the minority class
                    y_pred[np.isnan(y_pred)] = 0
                    metric=ImBinaryMetric(y_test,y_pre)
                    scores1.append([
                        metric.f1()
                        ,metric.MCC()
                        ,metric.aucroc(y_pred)
                        ,metric.gmean()
                    ])
                    print(metric.f1())
                    metric_conf_m.append(metric.conf_m)
                    del model
                print('ave_trainingrun_time:\t\t{:.2f}s'.format(np.mean(traintimes)))
                print('ave_testingrun_time:\t\t{:.2f}s'.format(np.mean(testtime)))
                print('------------------------------')
                print('Metrics:')
                print(metric_conf_m)
                df_scores = pd.DataFrame(scores1, columns=['F1', 'MMC', 'AUC', 'G_MEAN'])
                ssc = []
                for metric in df_scores.columns.tolist():
                    ssc.append(df_scores[metric].mean())
                scores.append(ssc)
            df_scores = pd.DataFrame(scores,columns=['F1', 'MMC', 'AUC', 'G_MEAN'])
            jilu=[]
            for metric in df_scores.columns.tolist():
                print ('{}\tmean:{} '.format(metric, df_scores[metric]))
                print ('{}\tmean:{:.3f}  std:{:.3f}'.format(metric, df_scores[metric].mean(), df_scores[metric].std()))
                jilu.append(df_scores[metric].mean())
                # jilu.append(df_scores[metric].std())
                # print (metric, '\t',df_scores[metric])
            jilu = np.array(jilu).reshape(1,4)
            # jilu = pd.DataFrame(jilu,columns=['F1', 'std1','MMC','std2', 'AUC', 'std3','G_MEAN','std4'])
            jilu = pd.DataFrame(jilu, columns=['F1', 'MMC', 'AUC', 'G_MEAN'])
            aaaa = aaaa.append(jilu, ignore_index=True)
        print(aaaa)
        for metric in aaaa.columns.tolist():
            print('{}\tmean:{:.3f} '.format(metric, aaaa[metric].mean()))
    return
if __name__ == '__main__':
    main()
            