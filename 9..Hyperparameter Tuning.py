!pip install scikit-learn=='0.24.1'
!pip install mlxtend=='0.18.0'
!pip install -U imbalanced-learn==0.8.0 
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import string
# Run in python console
import nltk; nltk.download('stopwords')
import re
from pprint import pprint

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from numpy import mean
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

#Imbalance Learn
from imblearn.ensemble import BalancedRandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import seaborn as sns

new_X = pd.read_csv('/content/drive/My Drive/PhD/JOM_data/new_X.csv').drop(['Unnamed: 0'], axis=1)
y = pd.read_csv('/content/drive/My Drive/PhD/JOM_data/y.csv').drop(['Unnamed: 0'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.3,
                                                    random_state = 5, stratify = y)

#Decision Tree

###############################################################################
#                               1. Classifiers                                 #
###############################################################################


estimator = DecisionTreeClassifier(
                              random_state = 5)
###############################################################################
#                             2. Hyper-parameters                             #
###############################################################################

# Update dict with Light GBM Parameters and always remember to put classifier__ with double underscore at each parameter


param_grid_cart = {"estimator__max_features": ["auto", "sqrt", "log2"
                                               ],
                   "estimator__criterion": ['gini','entropy'
                                            ],
                   "estimator__max_depth" : [13,14,15,16,17],
                   "estimator__min_samples_split": [4,5,6],
                   "estimator__min_samples_leaf": [3,4,5],
                   #'class_weight': class_weight_list
                   }

###############################################################################
#                     3. Tuning a classifier to use with RFECV                #
###############################################################################
# Define classifier to use as the base of the recursive feature elimination algorithm
selected_classifier = "DT"
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import RandomUnderSampler
under = RandomUnderSampler()
pipeline = Pipeline([
                     ('resampling',under), 
                     ('estimator', estimator)
      ])
# Create K-fold cross-validation 
from sklearn.model_selection import RepeatedKFold, KFold, StratifiedKFold, RepeatedStratifiedKFold
rtf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3 #, shuffle=True
                              )

# Initialize RandomizedGridSearch First object, only add estimator__ when put it into pipeline

gscv = GridSearchCV(pipeline, param_grid = param_grid_cart, cv = rtf, verbose = 100, scoring = "balanced_accuracy",refit=True, n_jobs=-1,
                    )
gscv = RandomizedSearchCV(pipeline, param_distributions = param_grid_cart, cv = rtf, verbose = 100, scoring = "balanced_accuracy",refit=True, n_jobs=-1)


                  
# Fit gscv
print(f"Now tuning {selected_classifier}. Go grab a beer or something.")
gscv.fit(X_train, y_train)

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_

# SVC

###############################################################################
#                               1. Classifiers                                 #
###############################################################################
classifiers = {}

estimator = LinearSVC(random_state=5, verbose=100)
###############################################################################
#                             2. Hyper-parameters                             #
###############################################################################
# Initiate parameter grid

# Update dict with Light GBM Parameters and always remember to put classifier__ with double underscore at each parameter

param_grid_rf = {"estimator__penalty": ['l1','l2'],
                 "estimator__loss": ['hinge','squared_hinge'],
                 "estimator__dual":[True,False],
                 "estimator__tol": [1e-5,1e-4,1e-3,1e-2,1e-1,1],
                 'estimator__C': [0.1, 1, 10, 100],
}

###############################################################################
#                     3. Tuning a classifier to use with RFECV                #
###############################################################################
# Define classifier to use as the base of the recursive feature elimination algorithm
selected_classifier = "SVC"
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import RandomOverSampler

over = RandomOverSampler()
pipeline = Pipeline([
                     ('resampling',over), 
                     ('estimator', estimator)
      ])
# Create K-fold cross-validation 
from sklearn.model_selection import RepeatedKFold, KFold
rtf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3 #, shuffle=True
                              )

# Initialize RandomizedGridSearch First object, only add estimator__ when put it into pipeline
gscv = RandomizedSearchCV(pipeline, param_distributions = param_grid_rf, cv = rtf, verbose = 100, scoring = "balanced_accuracy",refit=True,
                          random_state = 5)

gscv = GridSearchCV(pipeline, param_grid = param_grid_rf, cv = rtf, verbose = 100, scoring = "balanced_accuracy",refit=True, n_jobs=-1,
                    )
                  
# Fit gscv
gscv.fit(X_train, np.ravel(y_train.values))

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_

#LR

###############################################################################
#                               1. Classifiers                                 #
###############################################################################


estimator = LogisticRegression(random_state=5)
###############################################################################
#                             2. Hyper-parameters                             #
###############################################################################
# Initiate parameter grid

# Update dict with Light GBM Parameters and always remember to put classifier__ with double underscore at each parameter

param_grid_lr = {'estimator__penalty' : ['l1', 'l2'],
                 'estimator__C' : np.logspace(-4, 4, 30),
                 'estimator__solver' : ['liblinear','saga','sag','lbfgs'],
}

###############################################################################
#                     3. Tuning a classifier to use with RFECV                #
###############################################################################
# Define classifier to use as the base of the recursive feature elimination algorithm
selected_classifier = "Logistic Regression"

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import RandomOverSampler

over = RandomOverSampler()
pipeline = Pipeline([
                     ('resampling',over), 
                     ('estimator', estimator)
      ])
# Create K-fold cross-validation 
from sklearn.model_selection import RepeatedKFold, KFold, StratifiedKFold,RepeatedStratifiedKFold
rtf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3#, random_state= 5
                      )

# Initialize RandomizedGridSearch First object, only add estimator__ when put it into pipeline
gscv = RandomizedSearchCV(pipeline, param_distributions = param_grid_lr, cv = rtf,  n_jobs= -1, verbose = 100, scoring = 'balanced_accuracy' ,refit=True,
                         random_state = 5
                    )


gscv = GridSearchCV(pipeline, param_grid = param_grid_lr, cv = rtf, verbose = 100, scoring = 'balanced_accuracy',refit=True, n_jobs=-1)

# Fit gscv
print(f"Now tuning {selected_classifier}. Go grab a beer or something.")
gscv.fit(X_train, np.ravel(y_train.values))

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_

#NN

###############################################################################
#                               1. Classifiers                                 #
###############################################################################


estimator = MLPClassifier(random_state=5)
###############################################################################
#                             2. Hyper-parameters                             #
###############################################################################
# Initiate parameter grid

# Update dict with Light GBM Parameters and always remember to put classifier__ with double underscore at each parameter

param_grid_mlp = {'estimator__hidden_layer_sizes' : [(10,),(10,10),(10,10,10)],
                  'estimator__activation' : ['identity','logistic','tanh','relu'],
                  'estimator__solver' : ['lbfgs','sgd','adam'],
                  'estimator__alpha':[1e-1, 1e-2,1e-3, 1e-4, 1e-5],
                  'estimator__batch_size':[10, 25, 50, 100, 200],
                  'estimator__learning_rate':['constant','invscaling','adaptive'],
                  'estimator__learning_rate_init':[1e-1, 1e-2,1e-3,1,10,100],
                  'estimator__momentum':[0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
                  'estimator__tol':[1e-1, 1e-2,1e-3, 1e-4, 1e-5],

}

###############################################################################
#                     3. Tuning a classifier to use with RFECV                #
###############################################################################
# Define classifier to use as the base of the recursive feature elimination algorithm
selected_classifier = "Neural Networks"

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import RandomOverSampler

over = RandomOverSampler()
pipeline = Pipeline([
                     ('resampling',over), 
                     ('estimator', estimator)
      ]) 
# Create K-fold cross-validation 
from sklearn.model_selection import RepeatedKFold, KFold, StratifiedKFold,RepeatedStratifiedKFold
rtf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3#, random_state= 5
                      )

# Initialize RandomizedGridSearch First object, only add estimator__ when put it into pipeline
gscv = RandomizedSearchCV(pipeline, param_distributions = param_grid_mlp, cv = rtf,  n_jobs= -1, verbose = 100, scoring = 'balanced_accuracy' ,refit=True,
                        random_state = 5
                   )


gscv = GridSearchCV(pipeline, param_grid = param_grid_mlp, cv = rtf, verbose = 100, scoring = 'balanced_accuracy',refit=True, n_jobs=-1)

# Fit gscv
print(f"Now tuning {selected_classifier}. Go grab a beer or something.")
gscv.fit(X_train, np.ravel(y_train.values))

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_

#GB


###############################################################################
#                               1. Classifiers                                 #
###############################################################################

estimator = lgb.LGBMClassifier(random_state=5, max_iter=1000)

###############################################################################
#                             2. Hyper-parameters                             #
###############################################################################
# Initiate parameter grid
param_grid_lightgbm = {'depth':[3,1,2,6,4,5,7,8,9,10],
                       'iterations':[250,100,500,1000],
                       'learning_rate':[0.03,0.001,0.01,0.1,0.2,0.3], 
                       'l2_leaf_reg':[3,1,5,10,100],
                       'border_count':[32,5,10,20,50,100,200],
                       'ctr_border_count':[50,5,10,20,100,200],
                       'thread_count':[4],
                       'scale_pos_weight': [int(x) for x in np.linspace(5, 110, num = 22)],
                       'class_weight':[{0:1,1:2},{0:1,1:5},{0:1,1:10},{0:1,1:15},{0:1,1:20},{0:1,1:50},{0:1,1:100}],
                       'n_estimators':[int(x) for x in np.linspace(start = 200, stop = 8000, num = 10)]
                       }

###############################################################################
#                     2. Tuning a classifier to use with RFECV                #
###############################################################################
# Define classifier to use as the base of the recursive feature elimination algorithm
selected_classifier = "Light GBM"
# Create K-fold cross-validation 
from sklearn.model_selection import RepeatedKFold, KFold, StratifiedKFold, RepeatedStratifiedKFold
rtf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3 #, shuffle=True
                              )

# Initialize RandomizedGridSearch First object
gscv = RandomizedSearchCV(estimator, param_distributions = param_grid_lightgbm, cv = rtf,  n_jobs= -1, verbose = 100, scoring = "balanced_accuracy",refit=True,
                          random_state=5)

gscv = GridSearchCV(estimator, param_grid = param_grid_lightgbm, cv = rtf,  n_jobs= -1, verbose = 100, scoring = "balanced_accuracy",refit=True,
                   )

# Fit gscv
print(f"Now tuning {selected_classifier}. Go grab a beer or something.")
gscv.fit(X_train, np.ravel(y_train.values))

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_

# RF

###############################################################################
#                               1. Classifiers                                 #
###############################################################################

estimator = BalancedRandomForestClassifier(random_state=5)
###############################################################################
#                             2. Hyper-parameters                             #
###############################################################################
# Initiate parameter grid

# Update dict with Light GBM Parameters and always remember to put classifier__ with double underscore at each parameter

param_grid_rf = {"n_estimators": [int(x) for x in np.linspace(start = 200, stop = 8000, num = 10)],
                 "criterion": ['gini','entropy'],
                 "max_features": ["auto", "sqrt", "log2"],
                 "max_depth" : [int(x) for x in np.linspace(10, 110, num = 11)],
                 "min_samples_split": [2, 5, 7, 10, 15, 20, 50, 100],
                 "min_samples_leaf": [1, 2, 4, 5, 7, 10, 15, 20],
                 "bootstrap": [True, False],
                 "oob_score": [True, False],
                 'sampling_strategy': ['majority','not majority','all','auto']
}


###############################################################################
#                     2. Tuning a classifier to use with RFECV                #
###############################################################################
# Define classifier to use as the base of the recursive feature elimination algorithm
selected_classifier = "Balanced Random Forest"

# Create K-fold cross-validation 
from sklearn.model_selection import RepeatedKFold, KFold, StratifiedKFold, RepeatedStratifiedKFold
rtf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3,
                      #random_state= 5 , shuffle=True 
                      )

# Initialize RandomizedGridSearch First object, only add estimator__ when put it into pipeline
gscv = RandomizedSearchCV(estimator, param_distributions= param_grid_rf, cv = rtf, verbose = 100, scoring = "balanced_accuracy",refit=True, n_jobs=-1,
                          random_state = 5
                    )
gscv = GridSearchCV(estimator, param_grid = param_grid_rf, cv = rtf, verbose = 100, scoring = "balanced_accuracy",refit=True, n_jobs=-1,
                    )                  
# Fit gscv
print(f"Now tuning {selected_classifier}. Go grab a beer or something.")  
gscv.fit(X_train.values, np.ravel(y_train.values))

# Get best parameters and score
best_params = gscv.best_params_
best_score = gscv.best_score_

