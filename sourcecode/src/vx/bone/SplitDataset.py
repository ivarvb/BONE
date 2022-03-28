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

class SplitDataset:
    def __init__ (self, dat):
        self.dat = dat
        #self.evals = manager.list([{} for i in range(len(self.fetures)*len(self.dat["norms"]))])

    def execute(self):
        df = pd.read_csv(self.dat["csvfile"])
        odir = self.dat["outputdir"]
        classes = self.dat["classes"]
        outfile = self.dat["outfile"]
        
        
        dss = []
        for classe, values in classes.items():
            ds = []
            for subnames in values:
                dfx = df[df['target'] == subnames]
                #print(dfx)
                ds.append(dfx)
            
            dff = pd.concat(ds, ignore_index=True, sort=False)
            dff['target'] = classe

            dss.append(dff)

        dff = pd.concat(dss, ignore_index=True, sort=False)
        print(dff)
        dff.to_csv(odir+outfile, index=False)



if __name__ == "__main__": 
    dat =   {
                #"csvfile":"/mnt/sda6/software/frameworks/data/bone/build/csv/data.csv",
                #"csvfile":"/mnt/sda6/software/frameworks/data/bone/build/csv/data_v2.csv",
                "csvfile":"/mnt/sda6/software/frameworks/data/bone/build/csv/data_v3_r3.csv",
                "outputdir":"/mnt/sda6/software/frameworks/data/bone/build/datasets/",
                "classes":{
                    "SemFratura":["2-OsteopeniaSemFratura","4-OsteoporoseSemFratura"],
                    "ComFratura":["3-OsteopeniaComFratura","5-OsteoporoseComFratura"]
                },
                "outfile":"semcomfatura_v3.csv"
            }
    obje = SplitDataset(dat)
    obje.execute()





    dat =   {
                #"csvfile":"/mnt/sda6/software/frameworks/data/bone/build/csv/data.csv",
                #"csvfile":"/mnt/sda6/software/frameworks/data/bone/build/csv/data_v2.csv",
                "csvfile":"/mnt/sda6/software/frameworks/data/bone/build/csv/data_v3_r3.csv",
                "outputdir":"/mnt/sda6/software/frameworks/data/bone/build/datasets/",
                "classes":{
                    "osteopenia":["2-OsteopeniaSemFratura","3-OsteopeniaComFratura"],
                    "osteoporose":["4-OsteoporoseSemFratura","5-OsteoporoseComFratura"]
                },
                "outfile":"osteopenia_osteoporose_v3.csv"
            }
    obje = SplitDataset(dat)
    obje.execute()









