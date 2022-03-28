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
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

# tf.config.list_physical_devices('GPU')

# import os
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

##Variaveis globais
save_metrics_path = "../../../../data/bone/paper/Ivar/newexperiments/Metrics/"
#base_path_parts = "../../../../data/bone/paper/Jonathan/PartitionsFeatures/"
base_path_parts = "../../../../data/bone/paper/Ivar/newexperiments/PartitionsFeatures/"
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
    #train_Y, test_Y = list(), list()


    trainFrac = pd.read_csv(os.path.join(base_path_parts, "%.2d-train-fractures.csv"%(step)) )
    trainNoFrac = pd.read_csv(os.path.join(base_path_parts, "%.2d-train-nofractures.csv"%(step)) )
    #cc = list(trainFrac.columns)+list(trainNoFrac.columns)
    #print(cc)
    df_train = pd.concat([trainFrac,trainNoFrac], axis=0, ignore_index=True, sort=False)
    #print("df_train", df_train)
    #df_train.columns = cc
    columns = df_train.columns.tolist()
    #print("columns", columns)
    columns.remove("image")
    columns.remove("target")
    train_X = df_train[columns]
    train_Y = df_train.target.astype('category').cat.codes   
    #Xo = Xo.loc[:,Xo.apply(pd.Series.nunique) != 1]

    #classes = list(enumerate(df_train.target.astype('category').cat.categories))
    #classes = [dd[1] for dd in classes]
    #print("datcat train", classes)

    testFrac = pd.read_csv(os.path.join(base_path_parts, "%.2d-test-fractures.csv"%(step)))
    testNoFrac = pd.read_csv(os.path.join(base_path_parts, "%.2d-test-nofractures.csv"%(step)))
    df_test = pd.concat([testFrac,testNoFrac], axis=0, ignore_index=True, sort=False)
    columns = df_test.columns.tolist()
    #print("columnsx", columns)
    columns.remove("image")
    columns.remove("target")    
    test_X = df_test[columns]
    test_Y = df_test.target.astype('category').cat.codes   

    #classes = list(enumerate(df_test.target.astype('category').cat.categories))
    #classes = [dd[1] for dd in classes]
    #print("datcat test", classes)
    
    #print("train_X, test_X", train_X, test_X, train_Y, test_Y)

    return train_X, test_X, train_Y, test_Y

def load_balance_class_parts(mdl,step):
    train_X, test_X, train_Y, test_Y = load_dataset_part(step)
    #exit()
    #lb = LabelBinarizer()
    #train_Y = lb.fit_transform(train_Y)
    #test_Y = lb.fit_transform(test_Y)

    #sc = preprocessing.MinMaxScaler()
     
    ##Balanceameto dos dados de treino
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=7)
    train_X, train_Y = undersample.fit_resample(train_X, train_Y)

    #sc = preprocessing.MinMaxScaler()
    #sc = preprocessing.StandardScaler()
    #train_X = sc.fit_transform(train_X)
    

    ##Balanceamento dos dados de test
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=7)
    test_X, test_Y = undersample.fit_resample(test_X, test_Y)
    
    #sc = preprocessing.MinMaxScaler()
    #sc = preprocessing.StandardScaler()
    #test_X = sc.fit_transform(test_X)

    
    if mdl["norm"] == "std":
        sc = preprocessing.StandardScaler()
        train_X = sc.fit_transform(train_X)
        sc = preprocessing.StandardScaler()
        test_X = sc.fit_transform(test_X)
    

    return train_X, train_Y, test_X, test_Y 


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



    """
    learning_rate=0.1,
    feature_fraction=0.7, 
    scale_pos_weight=1.5,
    eval_metric='mlogloss', use_label_encoder=False """

    clfs = {        
        "RFC":{
            "model":RandomForestClassifier(n_estimators=400, random_state=7, n_jobs=-1),
            "norm":"none"
        },
        "XGBC":{
            "model":XGBClassifier(eval_metric='logloss', use_label_encoder=False),
            "norm":"none"
        },
        "GNBC":{
            "model":GaussianNB(),
            "norm":"std"
        },
        "DTC":{
            "model":DecisionTreeClassifier(random_state=7),
            "norm":"none"
        },
        "ADBC":{
            "model":AdaBoostClassifier(random_state=7),
            "norm":"std"
        },
        "KNNC":{
            "model":KNeighborsClassifier(n_neighbors = 2),
            "norm":"std"
        },
        "SVMC":{
            "model":svm.SVC(kernel="rbf", probability=True, C=10, gamma=0.001),
            "norm":"std"
        },

        # ########## GridSearchCV
        "RFC_GRID":{
            "model":GridSearchCV( 
                estimator = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1),
                param_grid = rf_parameters,
                #cv = 10,
                cv = cv,
                verbose=2,
                #scoring="roc_auc",
                scoring='accuracy',
                error_score=0,
                n_jobs=-1
            ),
            "norm":"none"
        },
    }
    return clfs


def evaluation(scores, y_true, y_pred):

    y_true, y_pred = y_true.tolist(), y_pred.tolist()

    acc = metrics.accuracy_score(y_true, y_pred, normalize=True)
    f1 = metrics.f1_score(y_true, y_pred)
    #roc_curve = metrics.roc_curve(y_true, y_pred)
    #roc_auc_score = [metrics.roc_auc_score(y_true, y_pred)]
    jac = metrics.jaccard_score(y_true, y_pred)
    pre = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
        
    scores["acc"].append(acc)
    scores["f1"].append(f1)
    #scores["roc_auc_score"].append(acc)
    scores["jac"].append(jac)
    scores["pre"].append(pre)
    scores["rec"].append(rec)



if __name__ == '__main__':

    # Simple
    modelnames = ["RFC","XGBC","KNNC","SVMC"]

    ## GridSearchCV
    #modelnames = ["RFC_GRID"]
        

    # para cada modelo
    for mdname in modelnames:
        for step in range(1,101,1):

            cls = classifiers()
            mdl = cls[mdname]
            save_name_train = mdname+"_train.csv"
            save_name_test = mdname+"_test.csv"

            print(f"Step: {step}")
            
            print("Load features")
            X_feat_train, train_under_Y, X_feat_test, test_under_Y = load_balance_class_parts(mdl, step)
            
            #print("X_feat_train.shape", X_feat_train.shape)
            
            print("Trainning %s"%(mdname))
            start_train = time.time()
            
            
            initClassifier = mdl["model"].fit(X_feat_train, train_under_Y.ravel())
            y_pred_train  = initClassifier.predict(X_feat_train)
            y_proba_train = initClassifier.predict_proba(X_feat_train)[:, 1]
            
            runtimeTrain = time.time() - start_train
            print("%s Trained in %2.2f seconds"%(mdname, runtimeTrain))


            print("Testing %s"%(mdname))
            start_test = time.time()
            y_pred_test = initClassifier.predict(X_feat_test)
            y_proba_test = initClassifier.predict_proba(X_feat_test)[:, 1]
            
            runtimeTest = time.time() - start_test
            print("%s Tested in %2.2f seconds"%(mdname, runtimeTest)) 

            #scores = {"acc":[],"f1":[],"jac":[],"pre":[],"rec":[]}
            #evaluation(scores, test_under_Y.ravel(), y_pred_test.ravel())
            #print ("scores", scores)
            
            calculateMeasures(y_pred_train, train_under_Y, y_proba_train, step, save_name_train)
            calculateMeasures(y_pred_test, test_under_Y, y_proba_test, step, save_name_test)
            #break

        
