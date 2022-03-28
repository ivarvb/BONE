#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Ivar
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler


from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression


from xgboost import XGBClassifier
import xgboost as xgb
from thundersvm import SVC

from Util import *
from multiprocessing import Pool, Manager, Process, Lock



from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#from sklearn.feature_selection import SelectKBest

from sklearn.svm import LinearSVC


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


from sklearn.feature_selection import GenericUnivariateSelect
from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_selection import RFECV

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LassoCV


from sklearn import datasets, linear_model
from genetic_selection import GeneticSelectionCV


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SequentialFeatureSelector


from sklearn.svm import SVR

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split


from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler


from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

#from pickle import dump, load

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import cross_validate


from sklearn.base import clone

from sklearn.linear_model import SGDClassifier

import time

class GDataset:
    def __init__ (self, dat):
        self.dat = dat

    def execute(self):
        df = pd.read_csv(self.dat["csvfile"])
        odir = self.dat["outputdir"]
        filterdir = self.dat["filterdir"]
        #print("df", df)
        

        for i in range(1, 101):
            df_filt = pd.read_csv(filterdir+str(i)+".csv")
            df_join = pd.concat([df,df_filt], axis=1, ignore_index=True, sort=False)
            cc = list(df.columns)+list(df_filt.columns)
            df_join.columns = cc

            df_join_train = df_join.loc[df_join['Trainning'] == 1]
            df_join_test = df_join.loc[df_join['Test'] == 1]

            df_join_train = df_join_train.drop(['Image', 'Trainning', 'Test'], axis=1)
            df_join_test = df_join_test.drop(['Image', 'Trainning', 'Test'], axis=1)


            df_join_train_sem_f = df_join_train.loc[df_join_train['target'] == 'SemFratura']
            df_join_train_com_f = df_join_train.loc[df_join_train['target'] == 'ComFratura']

            df_join_test_sem_f = df_join_test.loc[df_join_test['target'] == 'SemFratura']
            df_join_test_com_f = df_join_test.loc[df_join_test['target'] == 'ComFratura']

            #print("df_join_train_sem_f, df_join_train_com_f, df_join_test_sem_f, df_join_test_com_f", df_join_train_sem_f, df_join_train_com_f, df_join_test_sem_f, df_join_test_com_f)

            df_join_train_sem_f.to_csv("%s%.2d-train-nofractures.csv"%(odir, i), index=False)
            df_join_train_com_f.to_csv("%s%.2d-train-fractures.csv"%(odir, i), index=False)

            df_join_test_sem_f.to_csv("%s%.2d-test-nofractures.csv"%(odir, i), index=False)
            df_join_test_com_f.to_csv("%s%.2d-test-fractures.csv"%(odir, i), index=False)

            print("part %d"%(i) )

if __name__ == "__main__": 
    dat =   {
                #"csvfile":"/mnt/sda6/software/frameworks/data/bone/build/csv/data.csv",
                #"csvfile":"/mnt/sda6/software/frameworks/data/bone/build/csv/data_v2.csv",
                "csvfile":"/mnt/sda6/software/frameworks/data/bone/build/datasets/semcomfatura_v3.csv",
                "filterdir":"/mnt/sda6/software/frameworks/data/bone/paper/Jonathan/Partitions/",
                "outputdir":"/mnt/sda6/software/frameworks/data/bone/paper/Jonathan/PartitionsFeaturesIvar/",
            }

    obje = GDataset(dat)
    obje.execute()
