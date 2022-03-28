#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Ivar
"""

import sys
import os
#from scipy import interp
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

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier

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
from mlxtend.classifier import EnsembleVoteClassifier


from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

#from pickle import dump, load

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.model_selection import cross_validate


from sklearn.base import clone

from sklearn.linear_model import SGDClassifier

import time

import matplotlib.pyplot as plt

from sklearn.feature_selection import VarianceThreshold

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mlxtend.plotting import plot_confusion_matrix

class Classification:

    def __init__ (self, dat):
        self.dat = dat
        #self.evals = manager.list([{} for i in range(len(self.fetures)*len(self.dat["norms"]))])

    def execute(self):
        outdir = self.dat["outputdir"]+Util.now()
        Util.makedir(outdir)

        df = pd.read_csv(self.dat["csvfile"])
        columns = df.columns.tolist()
        print(df)
        print(columns)
        #exit()
        #prefixes = ('lbp-')
        #prefixes = ('LPB')
        prefixes = ()
        columns.remove("image")
        columns.remove("target")
        #print(columns)
        for word in columns[:]:
            if word.startswith(prefixes):
                columns.remove(word)


        """ 
        cc = 0
        for word in columns[:]:
            if word.startswith("log-"): #1023
            #if word.startswith("wavelet"): 372
            #if word.startswith("lbp-"):# 93
            #if word.startswith("LPB_"): 102
                cc+=1
        print("cc", cc)
        """


        classes = list(enumerate(df.target.astype('category').cat.categories))
        classes = [dd[1] for dd in classes]
        print("datcat", classes)


        evals = {}

        fig = plt.figure(figsize=(5,5))
        plt.title('')
        plt.plot([0, 1], [0, 1],'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')            
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])



        Xo = df[columns]
        Y = df.target.astype('category').cat.codes
        
        Xo = Xo.loc[:,Xo.apply(pd.Series.nunique) != 1]
        
        #Classification.featureSelection2(columns, Xo, Y, self.dat["classifiers"], outdir)

        cmdats = {}
        #iskfold = True
        #iskfold = False
        fpr, tpr = None, None
        for name, argms in self.dat["classifiers"].items():
            clsr =  Classification.classifiers()

            if name in clsr:
                print("len(columns)", len(columns))

                X = Xo.copy(deep=True)

                if argms["scale"] != "None":
                    scaler = Classification.getScale(argms["scale"])
                    X = pd.DataFrame(scaler.fit_transform(X))

                #sel = VarianceThreshold(threshold=0.12)
                #trainX = sel.fit_transform(trainX)

                trainX, testX, trainY, testY = train_test_split(
                            X, Y, stratify=Y, test_size=self.dat["testing"], random_state=7)

                """ 
                if argms["scale"] != "None":
                    scaler = Classification.getScale(argms["scale"])
                    trainX = pd.DataFrame(scaler.fit_transform(trainX))
                    testX = pd.DataFrame(scaler.transform(testX))
                """

                m = clsr[name]
                clf = m["model"]
                if len(argms["modelparameters"])>0:
                    clf = clf.set_params(**argms["modelparameters"])
                
                scores = Classification.evaluation_tmp()
                if self.dat["iskfold"]:
                    ytrue, ypred, fpr, tpr, roc_auc = Classification.kfolfcv(clf, X, Y, scores)
                    print("scoers", scores)
                    Classification.evaluationmean(scores)
                    evals[name]={"metrics":scores, "ytrue":ytrue, "ypred":ypred}
                    cmdats[name] = {"ytrue":ytrue, "ypred":ypred}
                    print(name,  evals[name]["metrics"])

                else:
                    #xx% train yy% test
                    #training
                    clf.fit(trainX, trainY)
                    #testing
                    pre = clf.predict(testX)
                
                    Classification.evaluation(scores, testY, pre)
                    Classification.evaluationmean(scores)
                    
                    evals[name]={"metrics":scores, "ytrue":testY.tolist(), "ypred":pre.tolist()}
                    cmdats[name] = {"ytrue":testY, "ypred":pre}
                    print(name,  evals[name]["metrics"])

                    #curver roc-auc
                    probs = clf.predict_proba(testX)
                    preds = probs[:,1]
                    fpr, tpr, threshold = metrics.roc_curve(testY, preds)
                    roc_auc = metrics.auc(fpr, tpr)
                    print("roc_auc", roc_auc)

                #plot auc-roc curves
                plt.plot(fpr, tpr, label = name+' (AUC) = %0.2f' % roc_auc)                
                
                #save best parameters
                if hasattr(clf, 'best_params_'):
                    print(name, clf.best_params_)

                #save fpr, tpr
                #FPR	TPR
                fprtprdf = pd.DataFrame({"FPR":fpr, "TPR":tpr})
                fprtprdf.to_csv(outdir+'/fpr_tpr_'+name+'.csv', index=False)
                del clf

        plt.legend(loc = 'lower right')
        fig.savefig(outdir+'/roc_auc.png', bbox_inches='tight')
        

        for key, val in cmdats.items():
            self.confmatrix(val["ytrue"], val["ypred"], classes, outdir+"/cm_"+key+".pdf", key)

        #print(evals)
        #self.evals[data["id"]] = evals
    @staticmethod
    def kfolfcv(model, X, y, scores):
        #auc
        tprs = []
        aucs = []
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)        



        #Implementing cross validation        
        k = 10
        #kf = KFold(n_splits=k, random_state=7)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=7)

        yltrue, ylpred = np.array([]),np.array([])
        for train_index , test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            
            Classification.evaluation(scores, y_test, y_pred)
            #print("kfolfcv", acc)
            #acc_score.append(acc)
            yltrue = np.concatenate((yltrue, y_test), axis=None)
            ylpred = np.concatenate((ylpred, y_pred), axis=None)



            probs = model.predict_proba(X_test)
            preds = probs[:,1]
            fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
            #tprs.append(interp(mean_fpr, fpr, tpr))
            

        mean_tpr /= float(k)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)

        return yltrue, ylpred, mean_fpr, mean_tpr, mean_auc
        #avg_acc_score = sum(acc_score)/k        






    @staticmethod
    def confmatrix(ytrue, ypred, classes, fo, title):
        #plt.title(title)

        cm = confusion_matrix(ytrue, ypred) 
        """ fig, ax = plot_confusion_matrix(conf_mat = cm,
                                        class_names = classes,
                                        show_absolute = True,
                                        show_normed = False,
                                        colorbar = True,
                                        title=title) """

        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot()

        plt.title(title)
        #plt.ylabel('True Label')
        #plt.xlabel('Predicated Label')
        
        #fig = plt.figure()        
        #fig.savefig(fo, bbox_inches='tight')
        plt.savefig(fo, bbox_inches='tight')


    @staticmethod
    def classifiers():
        """ 
        svm_parameters = [
            {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
            {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
        ] """

        svm_parameters = {
                        'C': [20],
                        'gamma': ['scale'],
                        'kernel': ['rbf'],
                        #'decision_function_shape': ['ovo'],
                    }
        xgb_parameters = {
            'max_depth': range (2, 7, 1),
            'n_estimators': range(250, 400, 50),
            'learning_rate': [0.1, 0.01, 0.05]
        }

        rf_parameters = {
            "n_estimators" : [10, 100, 1000],
            "max_features" : ['sqrt', 'log2']
        }

        gb_parameters = {
            "n_estimators" : [450],
            "learning_rate" : [0.001, 0.01, 0.1],
            "subsample" : [0.5, 0.7, 1.0]
        }
        

        cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=7)
        cv2 = ShuffleSplit(3, test_size=0.2, train_size=0.2, random_state=0)
        svm_parameters2 = [
            {"kernel": ["rbf"], "gamma": [0.1,0.01,0.001,0.0001,0.00001], "C": [1, 10, 100, 1000]},
            #{"kernel": ["linear"], "C": [1, 10, 100, 1000]},
        ]       

           
        classifiers = {
            "SVM_GRID":{
                "model":GridSearchCV(
                    estimator = svm.SVC(kernel='rbf', probability=True),
                    scoring='accuracy',
                    param_grid = svm_parameters2,
                    cv = cv,
                    verbose=2,
                    n_jobs=-1
                ),
            },
            "RFC":{"model":RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1)},
            "KNN":{"model":KNeighborsClassifier(n_neighbors = 3)},
            

            "SVMC":{"model":svm.SVC(kernel="rbf", probability=True, C=10, gamma=0.001)},
            
            "DTC":{"model":DecisionTreeClassifier(random_state=7)},
            "ADBC":{"model":AdaBoostClassifier(random_state=7)},
            "GNBC":{"model":GaussianNB()},


            "XGBC":{"model":XGBClassifier()},

            "GBC":{"model":GradientBoostingClassifier(
                learning_rate = 0.1,
                n_estimators = 450,
                subsample = 1.0)},

            "XGBCGRID":{"model":GridSearchCV(
                estimator=XGBClassifier(),
                param_grid=xgb_parameters,
                scoring='neg_log_loss',
                n_jobs = -1,
                cv = 5,
                verbose=True
            )},

            "SVMCGRID":{"model":GridSearchCV( 
                estimator = svm.SVC(probability=True),
                param_grid = svm_parameters,
                #cv = 10,
                cv = cv,
                verbose=2,
                #scoring="roc_auc",
                scoring='accuracy',
                error_score=0,
                n_jobs=-1)},

            "RFCGRID":{"model":GridSearchCV( 
                estimator = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1),
                param_grid = rf_parameters,
                #cv = 10,
                cv = cv,
                verbose=2,
                #scoring="roc_auc",
                scoring='accuracy',
                error_score=0,
                n_jobs=-1)},

            "GBCGRID":{"model":GridSearchCV( 
                estimator = GradientBoostingClassifier(),
                param_grid = gb_parameters,
                cv = cv,
                verbose=2,
                scoring="roc_auc",
                #scoring='accuracy',
                error_score=0,
                n_jobs=-1)},


            "ENSEM":{"model":EnsembleVoteClassifier(
                clfs=[
                    XGBClassifier(
                        learning_rate=0.1,
                        n_estimators=450,
                        max_depth=4,
                        min_child_weight=2,
                        gamma=0,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        nthread=5,
                        scale_pos_weight=1,
                        seed=27,
                        n_jobs=-1
                    ),
                    XGBClassifier(
                        learning_rate=0.1,
                        n_estimators=550,
                        max_depth=3,
                        min_child_weight=2,
                        gamma=0,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        nthread=5,
                        scale_pos_weight=1,
                        seed=27,
                        n_jobs=-1
                    ),
                    ],
                    weights=[2, 1], voting='soft')},



            "ETC":{"model":ExtraTreesClassifier(n_estimators=150, n_jobs=-1)},

            "ENSE":{"model":VotingClassifier(estimators=[AdaBoostClassifier(random_state=7), XGBClassifier()], voting='soft')},
            "SGDC":{"model":SGDClassifier(loss="hinge", penalty="l2", max_iter=5)},
            "MLPC":{"model":MLPClassifier(activation='tanh',solver='adam',alpha=1e-5,learning_rate='constant',random_state=7)},
        }

        return classifiers

    @staticmethod
    def get_voting():
        """
        clf_1 = KNeighborsClassifier()
        clf_2 = LogisticRegression()
        clf_3 = DecisionTreeClassifier()
        # Create voting classifier
        voting_ens = VotingClassifier(
        estimators=[('knn', clf_1), ('lr', clf_2), ('dt', clf_3)], voting='hard') """        

        models = list()
        models.append(('svm1', svm.SVC(probability=True, kernel='poly', degree=1)))
        models.append(('svm2', svm.SVC(probability=True, kernel='poly', degree=2)))
        models.append(('svm3', svm.SVC(probability=True, kernel='poly', degree=3)))
        models.append(('svm4', svm.SVC(probability=True, kernel='poly', degree=4)))
        models.append(('svm5', svm.SVC(probability=True, kernel='poly', degree=5)))
        
        #ensemble = VotingClassifier(estimators=models, voting='soft', n_jobs=-1)
        ensemble = VotingClassifier(estimators=models, voting='soft')
        return ensemble

    @staticmethod
    def evaluation_tmp():
        return {
            "acc":[],
            "f1":[],
            #"roc_auc_score":[],
            #"roc_auc":[],
            "jac":[],
            "pre":[],
            "rec":[],
            }

    @staticmethod
    def evaluation(scores, y_true, y_pred):

        y_true, y_pred = y_true.tolist(), y_pred.tolist()

        acc = metrics.accuracy_score(y_true, y_pred, normalize=True)
        f1 = metrics.f1_score(y_true, y_pred)
        #roc_curve = metrics.roc_curve(y_true, y_pred)
        #roc_auc_score = [metrics.roc_auc_score(y_true, y_pred)]
        jac = [metrics.jaccard_score(y_true, y_pred)]
        pre = [metrics.precision_score(y_true, y_pred)]
        rec = [metrics.recall_score(y_true, y_pred)]
        
        scores["acc"].append(acc)
        scores["f1"].append(f1)
        #scores["roc_auc_score"].append(acc)
        scores["jac"].append(jac)
        scores["pre"].append(pre)
        scores["rec"].append(rec)


    @staticmethod
    def evaluationmean(da):
        for k, v in da.items():
            da[k] = np.array(v).mean()
            
    @staticmethod
    def getScale(norm):
        sc = None
        if norm == "std":
            sc = StandardScaler()
        elif norm == "minmax":
            sc = MinMaxScaler()

        return sc

    @staticmethod
    def featureSelection(X, y):
        clsrfe =  Classification.classifiers()
        model = SFS(
                        #clsrfe["XGBC"]["model"],
                        clsrfe["RFC"]["model"],
                        k_features=180,
                        forward=True, 
                        floating=True, 
                        verbose=2,
                        scoring='f1',
                        cv=3,
                        n_jobs=-1
                        ).fit(X, y)
        selection = list(model.k_feature_idx_)
        print("selection", selection)


    @staticmethod
    def featureSelection2(columns, X, y, classifiers, outdir):       
        clsrfe =  Classification.classifiers()
        model = clsrfe["XGBC"]["model"]
        model = model.set_params(**classifiers["XGBC"]["modelparameters"])
        model.fit(X, y)

        columns = np.array(columns)
        sorted_idx = model.feature_importances_.argsort()
        columns[sorted_idx] = columns[sorted_idx]
        print(columns[sorted_idx])



        clsr =  Classification.classifiers()
        for name, argms in classifiers.items():
            if name in clsr:
                m = clsr[name]
                clf = m["model"]
                if len(argms["modelparameters"])>0:
                    clf = clf.set_params(**argms["modelparameters"])

                Xs = None
                for i in range(2,len(columns),2):
                    cols = columns[0:i]
                    Xs = X[cols]

                    scores = Classification.evaluation_tmp()
                    ytrue, ypred, fpr, tpr, roc_auc = Classification.kfolfcv(clf, Xs, y, scores)
                    dares = {"ytrue":ytrue.tolist(), "ypred":ypred.tolist(), "fpr":fpr.tolist(), "tpr":tpr.tolist(), "roc_auc":roc_auc, "scores":scores}
                    print("name, i, roc_auc, scores", name, i, roc_auc, scores)
                    Util.write(outdir+"/"+name+"_"+str(i)+".json", dares)


if __name__ == "__main__": 
    """ 
    dat =   {

                "outputdir":"../../../../data/bone/build/csv/results/featureselection/",
                #"csvfile":"../../data/bone/build/datasets/semcomfatura.csv",
                #"csvfile":"../../data/bone/build/datasets/semcomfatura_v2.csv",
                "csvfile":"../../../../data/bone/build/datasets/semcomfatura_v3.csv",
                #"csvfile":"../../data/bone/build/datasets/osteopenia_osteoporose_v2.csv",
                "testing":0.3,
                "iskfold":True,
                "classifiers":{

                                #"SVMC":{
                                #    "modelparameters":{
                                #        'C': 5, 'gamma': 'scale', 'kernel': 'rbf'},
                                #    "scale":"std",
                                #},

                                #"RFC":{
                                #    "modelparameters":{"n_estimators":450, "random_state":7, "n_jobs":-1},
                                #    "scale":"None",
                                #},

                                "XGBC":{
                                    "modelparameters":{
                                            "learning_rate":0.1,
                                            "n_estimators":450,
                                            "max_depth":4,
                                            "min_child_weight":2,
                                            "gamma":0,
                                            "subsample":0.8,
                                            "colsample_bytree":0.8,
                                            "objective":"binary:logistic",
                                            "nthread":5,
                                            "scale_pos_weight":1,
                                            "seed":7,
                                            "n_jobs":-1
                                    },
                                    "scale":"None",
                                },
                                
                                #"SVMCGRID":{
                                #    "modelparameters":{},
                                #    "scale":"std"
                                #},

                                #"RFCGRID":{
                                #    "modelparameters":{},
                                #    "scale":"std"
                                #},

                                #"GBCGRID":{
                                #    "modelparameters":{},
                                #    "scale":"std"
                                #},
                                
                                #"ENSEM":{
                                #    "modelparameters":{},
                                #    "scale":"None"
                                #},

                                #"GBC":{
                                #    "modelparameters":{},
                                #    "scale":"std"
                                #},
                                

                                ################'C': 10, 'gamma': 0.001, 'kernel': 'rbf'
                                
                                #"XGBCGRID":{"modelparameters":{}},
                                
                                #"ADBC":{"modelparameters":{}},
                                #"GNBC":{"modelparameters":{}},
                                #"DTC":{"modelparameters":{}},
                                #"MLPC":{"modelparameters":{}},
                                
                                #"SVCRBFGRID":{"modelparameters":{}},

                                #"MLPC":{"modelparameters":{
                                #            "activation":'tanh',
                                #            "solver":'adam',
                                #            "alpha":1e-5,
                                #            "learning_rate":'constant',
                                #            "random_state":7
                                #            }
                                #        }
                            }
            }

    obje = Classification(dat)
    obje.execute()








    """


    dat =   {
                "outputdir":"../../../../data/bone/build/csv/results/featureselection/",
                #"csvfile":"../../../../data/bone/ds2/build/csv/data_v1.csv",
                "csvfile":"../../../../data/bone/ds2/build/csv/data_v4.csv",
                "testing":0.2,
                "iskfold":False,
                "classifiers":{
                                "SVM_GRID":{
                                    "modelparameters":{},
                                    "scale":"std",
                                },
                                "SVMC":{
                                    "modelparameters":{
                                        #'C': 5, 'gamma': 'scale', 'kernel': 'rbf'
                                        "kernel":"rbf", "probability":True, "C":10, "gamma":0.001
                                    },
                                    "scale":"std",
                                },
                                "XGBC":{
                                    "modelparameters":{
                                            "learning_rate":0.1,
                                            "n_estimators":450,
                                            "max_depth":4,
                                            "min_child_weight":2,
                                            "gamma":0,
                                            "subsample":0.8,
                                            "colsample_bytree":0.8,
                                            "objective":"binary:logistic",
                                            "nthread":5,
                                            "scale_pos_weight":1,
                                            "seed":7,
                                            "n_jobs":-1
                                    },
                                    "scale":"None",
                                },
                            }
            }

    obje = Classification(dat)
    obje.execute()

