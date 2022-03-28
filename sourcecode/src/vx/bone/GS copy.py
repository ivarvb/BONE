import time
import os
#import tensorflow as tf
#from keras import backend as K
import numpy as np
from PIL import Image
import pandas as pd
import glob
import matplotlib.pyplot as plt
import csv
# from sklearn.model_selection import GridSearchCV
from tune_sklearn import TuneSearchCV
from tune_sklearn import TuneGridSearchCV
from sklearn.linear_model import SGDClassifier
#from ray import tune

from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
#pip install imbalanced-learn
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import subprocess
#from ray.tune.suggest.bayesopt import BayesOptSearch
from xgboost import XGBClassifier


from sklearn.metrics import auc, roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import datetime







from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
# tf.config.list_physical_devices('GPU')

# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##Variaveis globais
save_metrics_path = "../../../../data/bone/paper/Jonathan/Metrics/"
save_net_name_test = "bayes_test.csv"
save_net_name_test2 = "bayes_testKeras.csv"
save_net_name_train = "bayes_train.csv"
base_path_parts = "../../../../data/bone/paper/Jonathan/PartitionsFeatures/"
files_parts = os.listdir(base_path_parts)
input_size = (80,80)
runtimeTrain = 0.0
runtimeTest = 0.0



def specificity(tn, fp):
    return tn / (tn + fp)

# Negative Predictive Error
def npv(tn, fn):
    return tn / (tn + fn + 1e-7)

# Matthews Correlation_Coefficient
def mcc(tp, tn, fp, fn):
    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / np.sqrt(den + 1e-7)



"""
def calculateMeasuresTest(y_pred, y_true, scores, folder):
    metricsTest = pd.DataFrame()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    #fpr, tpr, _ = roc_curve(y_true, scores, pos_label=2)
    auc_val = roc_auc_score(y_true, scores)

    # Test RESULTS
    metricsTest['folder'] = [folder]
    metricsTest['accuracy'] = [accuracy_score(y_true, y_pred)]
    metricsTest['precision'] = [precision_score(y_true, y_pred)]
    metricsTest['sensitivity'] = [recall_score(y_true, y_pred)]
    metricsTest['specificity'] = [specificity(tn,fp)]
    metricsTest['fmeasure'] = [f1_score(y_true, y_pred)]
    metricsTest['npv'] = [npv(tn, fn)]
    metricsTest['mcc'] = [mcc(tp, tn, fp, fn)]
    metricsTest['auc'] = [auc_val]
    metricsTest['tn'] = [tn]
    metricsTest['fp'] = [fp]
    metricsTest['fn'] = [fn]
    metricsTest['tp'] = [tp]
    metricsTest['runtime'] = [runtimeTest]

    print(metricsTest)

    if os.path.exists(os.path.join(save_metrics_path, save_net_name_test)):
        metricsTest.to_csv(os.path.join(save_metrics_path, save_net_name_test), sep=',', mode='a', index=False, header=False)
    else:
        metricsTest.to_csv(os.path.join(save_metrics_path, save_net_name_test), sep=',', index=False)
"""
   
#Funções importantes
def calculateMeasures(y_pred, y_true, scores, folder, save_net_name):
    metricsTrain = pd.DataFrame()
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    #fpr, tpr, _ = roc_curve(y_true, scores, pos_label=2)
    auc_val = roc_auc_score(y_true, scores)

    # TRAIN RESULTS
    metricsTrain['folder'] = [folder]
    metricsTrain['accuracy'] = [accuracy_score(y_true, y_pred)]
    metricsTrain['precision'] = [precision_score(y_true, y_pred)]
    metricsTrain['sensitivity'] = [recall_score(y_true, y_pred)]
    metricsTrain['specificity'] = [specificity(tn,fp)]
    metricsTrain['fmeasure'] = [f1_score(y_true, y_pred)]
    metricsTrain['npv'] = [npv(tn, fn)]
    metricsTrain['mcc'] = [mcc(tp, tn, fp, fn)]
    metricsTrain['auc'] = [auc_val]
    metricsTrain['tn'] = [tn]
    metricsTrain['fp'] = [fp]
    metricsTrain['fn'] = [fn]
    metricsTrain['tp'] = [tp]
    metricsTrain['runtime'] = [runtimeTrain]

    print(metricsTrain)

    if os.path.exists(os.path.join(save_metrics_path, save_net_name)):
        metricsTrain.to_csv(os.path.join(save_metrics_path, save_net_name), sep=',', mode='a', index=False, header=False)
    else:
        metricsTrain.to_csv(os.path.join(save_metrics_path, save_net_name), sep=',', index=False)   


def load_dataset(base_path):
    imagens, labels = list(), list()
    classes = os.listdir(base_path)
    for c in classes:
        for p in glob.glob(os.path.join(base_path, c, '.csv')):
            imagens.append(p)
            labels.append(c)
    
    return np.asarray(imagens), labels

def load_dataset_part(step):
    train_Y, test_y = list(), list()

    trainFrac = pd.io.parsers.read_csv(os.path.join(base_path_parts, "%.2d-train-fractures.csv"%(step)), skiprows=1)
    for i in range(0,len(trainFrac), 1):
        train_Y.append('ComFratura')
    trainNoFrac = pd.io.parsers.read_csv(os.path.join(base_path_parts, "%.2d-train-nofractures.csv"%(step)), skiprows=1)
    for i in range(0,len(trainNoFrac), 1):
        train_Y.append('SemFratura')

    train_X = np.concatenate((trainFrac, trainNoFrac), axis=0)

    testFrac = pd.io.parsers.read_csv(os.path.join(base_path_parts, "%.2d-test-fractures.csv"%(step)), skiprows=1)
    for i in range(0,len(testFrac), 1):
        test_y.append('ComFratura')
    testNoFrac = pd.io.parsers.read_csv(os.path.join(base_path_parts, "%.2d-test-nofractures.csv"%(step)), skiprows=1)
    for i in range(0,len(testNoFrac), 1):
        test_y.append('SemFratura')

    test_x = np.concatenate((testFrac, testNoFrac), axis=0)
    
    return train_X, test_x, train_Y, test_y

def load_balance_class_parts(step):
    train_X, test_X, train_Y, test_Y = load_dataset_part(step)
    
    lb = LabelBinarizer()
    train_Y = lb.fit_transform(train_Y)
    test_Y = lb.fit_transform(test_Y)

    """  
    ##Balanceameto dos dados de treino
    undersample = RandomUnderSampler(sampling_strategy='majority')
    train_X, train_Y = undersample.fit_resample(train_X, train_Y)
    # print(train_under_Y)
    
    lb = LabelBinarizer()
    min_max_scaler = preprocessing.MinMaxScaler()
    # train_under_X = min_max_scaler.fit_transform(train_under_X)
    train_Y = lb.fit_transform(train_Y)
    

    ##Balanceamento dos dados de test
    undersample = RandomUnderSampler(sampling_strategy='majority')
    test_X, test_Y = undersample.fit_resample(test_X, test_Y)
    
    lb = LabelBinarizer()
    min_max_scaler = preprocessing.MinMaxScaler()
    test_Y = lb.fit_transform(test_Y)
    # test_under_X = min_max_scaler.fit_transform(test_under_X)
    """
   
    
    return train_X, train_Y, test_X, test_Y 



def train_ml_algorithm(tune_search, X_train, y_train):
    """ 
    clf = GaussianNB()
    parameter_grid = {
        'var_smoothing': np.logspace(0,-9, num=1000),
    }
    tune_search = GridSearchCV(clf,
        parameter_grid,n_jobs=-1, verbose=1, cv=2)
    seach_classfier = tune_search.fit(X_train, y_train.ravel()) """
    
    #print("y_train", y_train.ravel())
    print("len(X_train), len(y_train)", len(X_train), len(y_train))
    seach_classfier = tune_search.fit(X_train, y_train.ravel())
    return seach_classfier

def save_parts_proc(part):
    with open(os.path.join(save_metrics_path, "bayes/parts.txt"), mode="a") as f:
        f.write(f"{part}\n")
        
def load_parts_proc():
    parts = []
    with open(os.path.join(save_metrics_path, "bayes/parts.txt"), mode="r") as f:
        parts = f.readlines()

    parts = [p.replace("\n", "") for p in parts]
    
    return parts



def classifiers():
    cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=7)
    rf_parameters = {
            "n_estimators" : [10, 100, 1000],
            "max_features" : ['sqrt', 'log2']
    }

    clfs = {        
        "RFC":{
            "model":RandomForestClassifier(n_estimators=400, random_state=7, n_jobs=-1),
        },
        "RFCGRID":{
            "model":GridSearchCV( 
                estimator = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1),
                param_grid = rf_parameters,
                #cv = 10,
                cv = cv,
                verbose=2,
                #scoring="roc_auc",
                scoring='accuracy',
                error_score=0,
                n_jobs=-1)
        },
    }
    return clfs


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


def classification_i(csvfile):
    classifiersx = {"RFC":""}
    
    df = pd.read_csv(csvfile)
    columns = df.columns.tolist()
    #prefixes = ('lbp-')
    #prefixes = ('LPB')
    prefixes = ()
    columns.remove("image")
    columns.remove("target")
    #print(columns)
    for word in columns[:]:
        if word.startswith(prefixes):
            columns.remove(word)

    classes = list(enumerate(df.target.astype('category').cat.categories))
    classes = [dd[1] for dd in classes]
    print("datcat", classes)


    evals = {}
    
    """ 
    fig = plt.figure(figsize=(5,5))
    plt.title('')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')            
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01]) """

    Xo = df[columns]
    Y = df.target.astype('category').cat.codes
        
    Xo = Xo.loc[:,Xo.apply(pd.Series.nunique) != 1]
        
    cmdats = {}
    #iskfold = True
    #iskfold = False
    fpr, tpr = None, None
    for name, argms in classifiersx.items():
        
        clsr =  classifiers()
        m = clsr[name]
        clf = m["model"]

        if name in clsr:
            print("len(columns)", len(columns))

            X = Xo.copy(deep=True)

            """ 
            if argms["scale"] != "None":
                scaler = Classification.getScale(argms["scale"])
                X = pd.DataFrame(scaler.fit_transform(X)) """

            
            trainX, testX, trainY, testY = train_test_split(
                            X, Y,  stratify=Y, test_size=0.2, random_state=7)

            print("len(trainY), len(testY)", len(trainY), len(testY))

            #xx% train yy% test
            #training
            clf.fit(trainX, trainY)
            #testing
            pre = clf.predict(testX)
            
            scores = {
                "acc":[],
                "f1":[],
                #"roc_auc_score":[],
                #"roc_auc":[],
                "jac":[],
                "pre":[],
                "rec":[],
                }
            evaluation(scores, testY, pre)
            #evaluationmean(scores)
                    
            evals[name]={"metrics":scores, "ytrue":testY.tolist(), "ypred":pre.tolist()}
            cmdats[name] = {"ytrue":testY, "ypred":pre}
            print(name,  evals[name]["metrics"])

            #curver roc-auc
            probs = clf.predict_proba(testX)
            preds = probs[:,1]
            fpr, tpr, threshold = metrics.roc_curve(testY, preds)
            roc_auc = metrics.auc(fpr, tpr)
            print("roc_auc", roc_auc)


if __name__ == '__main__':
    """ 
    csvfile = "/mnt/sda6/software/frameworks/data/bone/build/datasets/semcomfatura_v3.csv"
    classification_i(csvfile)
    exit() """


    modelnames = ["RFC"]
    #modelnames = ["RFCGRID"]
    for mdname in modelnames:
        for step in range(1,101,1):
            # if folder in parts:
            #     continue

            cls = classifiers()
            mdl = cls[mdname]
            save_net_name_train = mdname+"_train.csv"
            save_net_name_test = mdname+"_test.csv"


            print(f"Step: {step}")
            
            print("Load features")
            X_feat_train, train_under_Y, X_feat_test, test_under_Y = load_balance_class_parts(step)
        
            
            print("Trainning %s"%(mdname))
            start_train = time.time()
            
            initClassifier = train_ml_algorithm(mdl["model"], X_feat_train, train_under_Y)
            y_pred_train  = initClassifier.predict(X_feat_train)
            y_proba_train = initClassifier.predict_proba(X_feat_train)[:, 1]
            
            runtimeTrain = time.time() - start_train
            print("%s Trained in %2.2f seconds"%(mdname, runtimeTrain))


            print("Testing %s"%(mdname))
            start_test = time.time()
            y_pred_test = initClassifier.predict(X_feat_test)
            y_proba_test = initClassifier.predict_proba(X_feat_test)[:, 1]
            
            print("len(X_test), len(y_test)", X_feat_train.shape, X_feat_test.shape, len(test_under_Y))

            runtimeTest = time.time() - start_test
            print("%s Tested in %2.2f seconds"%(mdname, runtimeTest)) 

            #print("y_pred_train", y_pred_train, y_pred_train.reshape(y_pred_train.shape[0], 1))
            scores = {"acc":[],"f1":[],"jac":[],"pre":[],"rec":[]}
            evaluation(scores, train_under_Y.ravel(), y_pred_train.ravel())
            print ("scores", scores)
            
            calculateMeasures(y_pred_train.reshape(y_pred_train.shape[0], 1), train_under_Y, y_proba_train, step, save_net_name_train)
            calculateMeasures(y_pred_test.reshape(y_pred_test.shape[0], 1), test_under_Y, y_proba_test, step, save_net_name_test)
            #break



        
