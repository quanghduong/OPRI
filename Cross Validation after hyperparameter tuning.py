#Classification model 
#Try Bagging Classification
# bagged decision trees on an imbalanced classification problem
from imblearn import FunctionSampler
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE, SMOTENC, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from numpy import mean
import sklearn
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, make_scorer, cohen_kappa_score
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_validate, cross_val_predict
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessClassifier
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.metrics import geometric_mean_score
from tqdm import tqdm 
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, confusion_matrix, plot_confusion_matrix, f1_score, recall_score, precision_score
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier,RUSBoostClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.3,
                                                    random_state = 5, stratify = y)

param = {'DecisionTreeClassifier':{'criterion': 'gini', 'max_depth': 16, 'max_features': 'auto', 'min_samples_leaf': 5, 'min_samples_split': 5},
         'LinearSVC':{'C': 10, 'dual': False, 'loss': 'squared_hinge', 'penalty': 'l2', 'tol': 4e-05},
         'BalancedRandomForestClassifier':{'criterion': 'entropy', 'max_depth': 80, 'max_features': 'auto', 'min_samples_leaf': 6, 'min_samples_split': 6, 'n_estimators': 1100, 'sampling_strategy': 'all'},
         'LogisticRegression':{'C': 425, 'penalty': 'l1', 'solver': 'liblinear'},
         'LGBMClassifier':{'border_count': 200, 'class_weight': {0: 1, 1: 5}, 'ctr_border_count': 50, 'depth': 9, 'iterations': 250, 'l2_leaf_reg': 3, 'learning_rate': 0.01, 'n_estimators': 3666, 'scale_pos_weight': 60, 'thread_count': 4},
         'BalancedBaggingClassifier':{'base_estimator': DecisionTreeClassifier(max_depth=10), 'bootstrap': False, 'bootstrap_features': False, 'max_features': 0.8, 'max_samples': 0.9, 'n_estimators': 2700, 'n_jobs': -1, 'oob_score': False, 'replacement': False, 'sampling_strategy': 'all', 'verbose': 100, 'warm_start': False},
         'MLPClassifier':{'activation': 'logistic', 'alpha': 2e-05, 'batch_size': 50, 'hidden_layer_sizes': (10, 15, 20), 'learning_rate': 'invscaling', 'learning_rate_init': 0.001, 'momentum': 0.3, 'solver': 'adam', 'tol': 0.02}} 
#model.__class__.__name__ 
over = RandomOverSampler(random_state=5)
under = RandomUnderSampler(random_state=5)
df_score = pd.DataFrame()
df_average = pd.DataFrame()

models = [DecisionTreeClassifier(random_state = 5), LinearSVC(random_state=5),BalancedRandomForestClassifier(random_state=5),
          LogisticRegression(random_state=5),lgb.LGBMClassifier(random_state=5),BalancedBaggingClassifier(random_state=5),
          MLPClassifier(random_state=5)]
y_train = np.ravel(y_train.values)



# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=5)

# evaluate model

metrics = {'balanced accuracy':'balanced_accuracy'}

for X_train, X_test, name in tqdm(zip([X_train],[X_test], ['All Variables'])):
  for model in tqdm(models):
    model.set_params(**param[model.__class__.__name__ ])
    if model.__class__.__name__ == 'DecisionTreeClassifier':
      pipeline = Pipeline([('under',under),
                          ('estimator', model)
                          ])
      scores = cross_validate(pipeline, X_train, y_train, scoring= metrics, cv=cv, n_jobs=-1, verbose=100)
      average_scores = pd.DataFrame()
      pipeline.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      recall = recall_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred)
      G_mean = geometric_mean_score(y_test, y_pred)
      balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred)
      kappa = cohen_kappa_score(y_test, y_pred)
      val = [recall, precision, G_mean, balanced_accuracy, f1,kappa]

      #combine
      average_scores = pd.concat([average_scores,pd.DataFrame(val, index=['recall','precision','g-mean','balanced_accuracy','f1','kappa'])]).T
      average_scores['Model'] = f'{model.__class__.__name__}'
      average_scores['Group'] = name
      scores['Model'] = f'{model.__class__.__name__}'
      scores['Group'] = name
      df_average = pd.concat([df_average, average_scores], axis=0)
      df_score = pd.concat([df_score, pd.DataFrame(scores)], axis=0)

    elif model.__class__.__name__ in 'LGBMClassifierBalancedRandomForestClassifierBalancedBaggingClassifier':
      pipeline = Pipeline([
                          ('estimator', model)
                          ])
      scores = cross_validate(pipeline, X_train, y_train, scoring= metrics, cv=cv, n_jobs=-1, verbose=100)
      average_scores = pd.DataFrame()
      pipeline.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      recall = recall_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred)
      G_mean = geometric_mean_score(y_test, y_pred)
      balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred)
      kappa = cohen_kappa_score(y_test, y_pred)
      val = [recall, precision, G_mean, balanced_accuracy, f1,kappa]

      #combine
      average_scores = pd.concat([average_scores,pd.DataFrame(val, index=['recall','precision','g-mean','balanced_accuracy','f1','kappa'])]).T
      if model.__class__.__name__ == 'BalancedRandomForestClassifier':
        average_scores['Model'] = 'RandomForestClassifier'
        average_scores['Group'] = name
        scores['Group'] = name
        scores['Model'] = 'RandomForestClassifier'
      elif model.__class__.__name__ == 'BalancedBaggingClassifier':
        average_scores['Model'] = 'BaggingClassifier'
        scores['Model'] = 'BaggingClassifier'
        average_scores['Group'] = name
        scores['Group'] = name
      else:
        average_scores['Model'] = f'{model.__class__.__name__}'
        scores['Model'] = f'{model.__class__.__name__}'
        average_scores['Group'] = name
        scores['Group'] = name
      df_average = pd.concat([df_average, average_scores], axis=0)
      df_score = pd.concat([df_score, pd.DataFrame(scores)], axis=0)


    else:
      pipeline = Pipeline([
                          ('over',over),
                          ('estimator', model)
                          ])
      scores = cross_validate(pipeline, X_train, y_train, scoring= metrics, cv=cv, n_jobs=-1, verbose=100)
      average_scores = pd.DataFrame()
      pipeline.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      recall = recall_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred)
      G_mean = geometric_mean_score(y_test, y_pred)
      balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred)
      kappa = cohen_kappa_score(y_test, y_pred)
      val = [recall, precision, G_mean, balanced_accuracy, f1,kappa]

      #combine
      average_scores = pd.concat([average_scores,pd.DataFrame(val, index=['recall','precision','g-mean','balanced_accuracy','f1','kappa'])]).T

      if model.__class__.__name__ == 'LinearSVC':
        average_scores['Model'] = 'SVC'
        scores['Model'] = 'SVC'
        average_scores['Group'] = name
        scores['Group'] = name
      else:
        average_scores['Model'] = f'{model.__class__.__name__}'
        scores['Model'] = f'{model.__class__.__name__}'
        average_scores['Group'] = name
        scores['Group'] = name
      df_average = pd.concat([df_average, average_scores], axis=0)
      df_score = pd.concat([df_score, pd.DataFrame(scores)], axis=0)

df_score = df_score.reset_index(drop=True)
df_average = df_average.reset_index(drop=True)


#Visualise results in boxplots
df_score = df_score.drop(['fit_time','score_time'],axis=1)
df_score['Group'] = 'All Variables'
df_average['Group'] = 'All Variables'
df_score.columns = ['Balanced Accuracy','Model','Group']
df_average.columns = ['Test Balanced Accuracy','Model','Group']
df_score['Model'] = df_score['Model'].replace(['LGBMClassifier','SVC','DecisionTreeClassifier','LogisticRegression','MLPClassifier',
                                                 'RandomForestClassifier'],
                                                ['Gradient Boosting','Support Vector Machine','Decision Tree','Logistic Regression','Neural Networks',
                                                 'Random Forest'])
df_average['Model'] = df_average['Model'].replace(['LGBMClassifier','SVC','DecisionTreeClassifier','LogisticRegression','MLPClassifier',
                                                 'RandomForestClassifier'],
                                                ['Gradient Boosting','Support Vector Machine','Decision Tree','Logistic Regression','Neural Networks',
                                                 'Random Forest'])
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator
import numpy as np
import regex as re
from matplotlib.patches import PathPatch
import matplotlib.ticker as ticker
def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def draw_boxplot(metric1, metric2, df1, df2):
  font = {'fontname':'Times New Roman'}
  colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
  length = len(df2)
  fig, ax = plt.subplots(figsize=(10,8))
  sns.boxplot(x= df1['Model'], y=df1[f'{metric1}']#,hue=df1['Group']
              ,  orient="v",dodge=True, data=df1, 
              )
  #sns.despine(offset=20, trim=False)
  adjust_box_widths(fig, 0.5)

  ax.xaxis.set_minor_locator(MultipleLocator(0.5))

  plt.yticks(np.arange(min(df1[f'{metric1}']), max(df1[f'{metric1}']), 0.85))
  ax.xaxis.grid(True, which='minor', color='black', lw=2, linestyle='--')

  #add test set
  plt.scatter(0, df2[f'{metric2}'][0],c='red',s=50, label ='Test set')
  plt.scatter(1,df2[f'{metric2}'][1],c='red',s=50)
  plt.scatter(2,df2[f'{metric2}'][2],c='red',s=50)
  plt.scatter(3,df2[f'{metric2}'][3],c='red',s=50)
  plt.scatter(4,df2[f'{metric2}'][4],c='red',s=50)
  plt.scatter(5,df2[f'{metric2}'][5],c='red',s=50)

  ax.legend(loc="upper right", prop={'size':12})
  plt.ylabel('Balanced Accuracy', size=20#, fontweight='bold'
            , **font)
  plt.xlabel('Model',size=20#, fontweight='bold'
            , **font)
  xlabels = sorted(list(df_score.Model.value_counts().index))

  xlabels_new = [re.sub("(\s)", "\\1\n", label, 0, re.DOTALL) for label in xlabels]

  plt.xticks(range(6), xlabels_new, size = 16, **font)

  plt.xticks(size = 14, **font)
  plt.yticks(size = 11, **font)
  start, end = ax.get_ylim()
  ax.yaxis.set_ticks(np.arange(start, end, 0.03))
  ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))
  plt.show()
draw_boxplot('Balanced Accuracy','Test Balanced Accuracy', df_score.sort_values(by=['Model']).reset_index(drop=True), df_average.sort_values(by=['Model']).reset_index(drop=True))import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator
import numpy as np
import regex as re
from matplotlib.patches import PathPatch
import matplotlib.ticker as ticker
def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def draw_boxplot(metric1, metric2, df1, df2):
  font = {'fontname':'Times New Roman'}
  colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
  length = len(df2)
  fig, ax = plt.subplots(figsize=(10,8))
  sns.boxplot(x= df1['Model'], y=df1[f'{metric1}']#,hue=df1['Group']
              ,  orient="v",dodge=True, data=df1, 
              )
  #sns.despine(offset=20, trim=False)
  adjust_box_widths(fig, 0.5)

  ax.xaxis.set_minor_locator(MultipleLocator(0.5))

  plt.yticks(np.arange(min(df1[f'{metric1}']), max(df1[f'{metric1}']), 0.85))
  ax.xaxis.grid(True, which='minor', color='black', lw=2, linestyle='--')

  #add test set
  plt.scatter(0, df2[f'{metric2}'][0],c='red',s=50, label ='Test set')
  plt.scatter(1,df2[f'{metric2}'][1],c='red',s=50)
  plt.scatter(2,df2[f'{metric2}'][2],c='red',s=50)
  plt.scatter(3,df2[f'{metric2}'][3],c='red',s=50)
  plt.scatter(4,df2[f'{metric2}'][4],c='red',s=50)
  plt.scatter(5,df2[f'{metric2}'][5],c='red',s=50)

  ax.legend(loc="upper right", prop={'size':12})
  plt.ylabel('Balanced Accuracy', size=20#, fontweight='bold'
            , **font)
  plt.xlabel('Model',size=20#, fontweight='bold'
            , **font)
  xlabels = sorted(list(df_score.Model.value_counts().index))

  xlabels_new = [re.sub("(\s)", "\\1\n", label, 0, re.DOTALL) for label in xlabels]

  plt.xticks(range(6), xlabels_new, size = 16, **font)

  plt.xticks(size = 14, **font)
  plt.yticks(size = 11, **font)
  start, end = ax.get_ylim()
  ax.yaxis.set_ticks(np.arange(start, end, 0.03))
  ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.4f'))
  plt.show()
draw_boxplot('Balanced Accuracy','Test Balanced Accuracy', df_score.sort_values(by=['Model']).reset_index(drop=True), df_average.sort_values(by=['Model']).reset_index(drop=True))

