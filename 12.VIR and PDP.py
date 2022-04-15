#VIR

%%time 
import shap
# explain the model's predictions on test set using SHAP values
# same syntax works for xgboost, LightGBM, CatBoost, and some scikit-learn models
explainer = shap.TreeExplainer(model)

#For all model
%%time
# shap_values consists of a list of two matrices of dimension samplesize x #features
# The first matrix uses average nr of begin samples as base value
# The second matrix which is used below uses average nr of malignant samples as base value 
shap_values = explainer.shap_values(X_train)
shap_values_bar = explainer(X_train)
#shapley values is the difference between actual prediction (Actual values) of an instance with the average prediction of all instances (predictions of each features)

import matplotlib.pyplot as plt

shap.plots.bar(shap_values_bar_c[:,:,1], max_display=16,show=False)
for i in ['Electronic Device','Food & Grocery','Fashion Item','Book','Home Appliance']:
  plt.xlabel('{}'.format(i),size=15#, fontweight='bold'
            )
  plt.show()

import matplotlib.pyplot as plt
import numpy as np

t = range(15)

fig = plt.figure(figsize=(40, 10))
fig.tight_layout(pad=10)
gs = fig.add_gridspec(5, hspace=0.1, ncols=1)


axs[0].spines["bottom"].set_visible(True)
axs[0].spines["top"].set_visible(False)
axs[0].spines["left"].set_visible(False)
axs[0].spines["right"].set_visible(False)

axs = gs.subplots(sharex=True, sharey=True)

axs[2].set_ylabel( 'Variable Important Ranking',va='center', rotation='vertical', fontsize=40, loc = "top")
axs[2].yaxis.set_label_coords(-0.01, 1.5)



l0 =axs[0].bar(t, data2['Book'].values, color='red',label='Book',width=0.7, align='center')
plt.xticks([],[])
for i,n in zip(range(15), data2['Book'].values):
  axs[0].vlines(x=i,color='grey',linewidth=1, ymin=n, ymax=16, linestyles='dotted') #{'solid', 'dashed', 'dashdot', 'dotted'}


l1 = axs[1].bar(t, data2['Fashion Item'].values, color = 'orange',label='Fahsion Item',width=0.7, align='center')
plt.xticks([],[])
for i,n in zip(range(15), data2['Fashion Item'].values):
  axs[1].vlines(x=i,color='grey',linewidth=1, ymin=n, ymax=16, linestyles='dotted') #{'solid', 'dashed', 'dashdot', 'dotted'}

l2 = axs[2].bar(t, data2['Food & Grocery'].values, color = 'blue',label = 'Food & Grocery',width=0.7, align='center')
plt.xticks([],[])
for i,n in zip(range(15), data2['Food & Grocery'].values):
  axs[2].vlines(x=i,color='grey',linewidth=1, ymin=n, ymax=16, linestyles='dotted') #{'solid', 'dashed', 'dashdot', 'dotted'}

l3 = axs[3].bar(t, data2['Home Appliance'].values, color = 'green', label='Home Appliance',width=0.7, align='center')
plt.xticks([],[])
for i,n in zip(range(15), data2['Home Appliance'].values):
  axs[3].vlines(x=i,color='grey',linewidth=1, ymin=n, ymax=16, linestyles='dotted') #{'solid', 'dashed', 'dashdot', 'dotted'}

l4 = axs[4].bar(t, data2['Electronic Device'].values, color ='gray', label='Electronic Device',width=0.7, align='center')
for i,n in zip(range(15), data2['Electronic Device'].values):
  axs[4].vlines(x=i,color='grey',linewidth=1, ymin=n, ymax=16, linestyles='dotted') #{'solid', 'dashed', 'dashdot', 'dotted'}


#Label on top
for index, value1, value2 in zip(range(15), data['Book'].values, data2['Book'].values):
  axs[0].text(index, value2, str(value1), fontsize=25, linespacing=1.5,ha='center', va='baseline')
for index, value1, value2 in zip(range(15), data['Fashion Item'].values, data2['Fashion Item'].values):
  axs[1].text(index, value2, str(value1), fontsize=25, linespacing=1.5,ha='center', va='baseline')
for index, value1, value2 in zip(range(15), data['Food & Grocery'].values, data2['Food & Grocery'].values):
  axs[2].text(index, value2, str(value1), fontsize=25, linespacing=1.5,ha='center', va='baseline')
for index, value1, value2 in zip(range(15), data['Home Appliance'].values, data2['Home Appliance'].values):
  axs[3].text(index, value2, str(value1), fontsize=25, linespacing=1.5,ha='center', va='baseline')
for index, value1, value2 in zip(range(15), data['Electronic Device'].values, data2['Electronic Device'].values):
  axs[4].text(index, value2, str(value1), fontsize=25, linespacing=1.5,ha='center', va='baseline')

import regex as re 

#xlabels_new = [re.sub("(\s)", "\\1\n", label, 0, re.DOTALL) for label in data.index]

plt.xticks(t, data.index, rotation='vertical', fontsize=40)

plt.yticks([],[])
# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=2, 
                    wspace=1.5, 
                    hspace=1.5)

plt.legend([l0, l1, l2, l3, l4], [ 'Book', 'Fashion Item','Food & Grocery',
        'Home Appliance','Electronic Device'], loc='upper center', bbox_to_anchor=(0.5, -2.5),
          fancybox=True, shadow=True, ncol=5, prop={'size': 30})
plt.rcParams['axes.linewidth'] = 0.01


plt.show()

#PDP

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size = 0.3,
                                                    random_state = 5, stratify = y)


from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, EasyEnsembleClassifier,RUSBoostClassifier
model = BalancedRandomForestClassifier(random_state=5)
param = {'criterion': 'entropy', 'max_depth': 80, 'max_features': 'auto', 'min_samples_leaf': 6, 'min_samples_split': 6, 'n_estimators': 1100, 'sampling_strategy': 'all'}
model.set_params(**param)

model.fit(X_train, np.ravel(y_train.values))


from sklearn.inspection import plot_partial_dependence
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from pdpbox import pdp, get_dataset, info_plots
import matplotlib.ticker as ticker
f = X_train[X_train['Product Category'] == 3]
c = X_train[X_train['Product Category'] == 1]
b = X_train[X_train['Product Category'] == 0]
e = X_train[X_train['Product Category'] == 2]
h = X_train[X_train['Product Category'] == 4]


'''one-way PDP'''
cats = [f, b, c, e, h
        ]
colors = ["red","grey","yellow","cyan","green",
          ]
cat_names = ["Food & Grocery","Book","Fashion Item","Electronic Device","Home Appliance"
             ]

font = {'fontname':'Times New Roman'}


def plot_pdp(model, cats, colors, cat_names, feature_name, limit, i, j, width):
  fig, ax = plt.subplots(figsize=(5,7))
  for cat, color, cat_name,  in zip(cats, colors, cat_names):
    ind = list(cat.columns).index(feature_name)
    if cat_name == 'Food & Grocery':
      disp = plot_partial_dependence(model, cat, [ind], line_kw={"label": cat_name,"color": color}, target=1, ax=ax, percentiles=(0,1))

    else: 
      disp = plot_partial_dependence(model, cat, [ind], line_kw={"label": cat_name,"color": color}, target=1, ax= disp.axes_, percentiles=(0,1))
  

  plt.vlines(x=i, ymin=0, ymax = 1, color='grey', linestyle = '--', lw=width)
  #for x, y in zip(i, j):
    #plt.plot(x, y, marker = 'o', markersize=1, color='grey')
  xt = disp.axes_[0][0].get_xticks()
  xt = np.append(xt,i[0])
  xtl=xt.tolist()
  xtl[-1] = f"~{xtl[-1]}"
  ax.set_xticks(xt)
  ax.set_xticklabels(xtl)
  plt.yticks( 
      size = 10
      )
  plt.xticks( 
      size = 10
      )
  plt.legend(loc="upper right", prop={'size':8})    
  plt.ylim(limit)
  plt.xlabel(feature_name, size=14)
  plt.ylabel('OPRI Probability',size=14)
  plt.show()

i = [51.8181818181818, 51.3838383838384, 51.8181818181818, 50.9090909090909, 51.7676767676768, 51.1515151515151, 130, 129.565656565657, 130, 130.393939393939, 131.545454545455, 131.393939393939]
j = [0.553801340016104, 0.493902110767097, 0.335514010751763, 0.556671845085714,  0.601609847575656, 0.577351106155739, 0.644195703863454, 0.598986322527732, 0.491717096881069, 0.647333564077438, 0.674373788690761, 0.66781577258337]
feature_name = 'Review Length'
limit = [0.1,0.85]
plot_pdp(model, cats, colors, cat_names, feature_name, limit, i, j, width=0.1)


i = [0.124144857766245, 0.114595253322688, 0.381984177742294, 0.381984177742294]
j = [0.599378503002044, 0.601674281376752, 0.714238705296242, 0.754364426757463]
feature_name = 'Return Management'
limit = [0.3,0.9]
plot_pdp(model, cats, colors, cat_names, feature_name, limit, i, j, width= 0.3)

i = [0.058132137600079, 0.058132137600079, 0.0585905962605473, 0.0588741924784699]
j = [0.359567440726666, 0.238385504492882, 0.371649585441481, 0.384338634294776]
feature_name = 'Primary Features'
limit = [0.15,0.45]
plot_pdp(model, cats, colors, cat_names, feature_name, limit, i, j, width=0.15)

i = [0.0831492747369774, 0.0831492747369774, 0.0840476596033319, 0.0834294004404329]
j = [0.511736159398464, 0.431257607503973, 0.445164906583819, 0.628701407589425]
feature_name = 'Packaging'
limit = [0.2,0.8]
plot_pdp(model, cats, colors, cat_names, feature_name, limit, i , j, width=0.15)

i = [0.00789573324767355, 0.00767925200923061, 0.00789573324767355, 0.114488132091266, 0.113268967136152,0.112514198779348, 0.155940731641553, 0.15550485318692, 0.155940731641553]
j = [0.386133902012975, 0.386674915319571, 0.448350310726084, 0.358030880645669, 0.356128256597864, 0.404651360846,0.31466378036887, 0.313048443284421, 0.339536856024018]
feature_name = 'Secondary Features'
limit = [0.2,0.5]
plot_pdp(model, cats, colors, cat_names, feature_name, limit, i, j, width=0.15)

i = [0.2235616851694, 0.2235616851694]
j = [0.603989727531474, 0.664816100165905]
feature_name = 'Customer Service'
limit = [0.3,0.9]
plot_pdp(model, cats, colors, cat_names, feature_name, limit, i, j, width=0.2)





i = [0.0116001100424292, 0.0114936036377934, 0.0116001100424292, 0.0110653595334164, 0.0111393200082215, 0.0111912643834286, 0.0629720259446154, 0.0623938483194497, 0.0629720259446154, 0.0622426473754671,
     0.062061925760091, 0.0623513301362453]
j = [0.36849936435026, 0.321750426306044, 0.235751722215785, 0.359180849707617, 0.400349237818488, 0.394518336910977, 0.360335271751678, 0.318550632695583, 0.251289419596178, 0.347936301005165, 
     0.394362480720017, 0.379116462049013]
feature_name = 'Reliability'
limit = [0.2,0.47]
plot_pdp(model, cats, colors, cat_names, feature_name, limit, i ,i, width = 0.1)

i = [-0.454545454545454, -0.454545454545454, -0.454545454545454, -0.454545454545454, -0.454545454545454, -0.454545454545454, -0.0101010101010101, -0.0101010101010101, -0.0101010101010101,
     -0.0101010101010101, -0.0101010101010101, -0.0303030303030303, 0.373737373737374, 0.373737373737374, 0.373737373737374, 0.373737373737374,0.373737373737374,0.373737373737374]
j = [0.395841955500702, 0.342574048437555, 0.253053478753028,0.419700105331153, 0.429666162675215, 0.404641205689236, 0.395581888724754, 0.319691212864811, 0.242714943166058,
     0.410056505540836, 0.432489821686901, 0.416154987445775, 0.351081143645896, 0.292648222955171, 0.204610481141648,0.356760413902291, 0.379092872247314, 0.378013086225379]
feature_name = 'Review Sentiment'
limit = [0.2,0.45]
plot_pdp(model, cats, colors, cat_names, feature_name, limit, i, j ,width = 0.1)




def plot_pdp_(model, cats, colors, cat_names, feature_name, limit):
  fig, ax = plt.subplots(figsize=(5,7))
  for cat, color, cat_name,  in zip(cats, colors, cat_names):
    ind = list(cat.columns).index(feature_name)
    if cat_name == 'Food & Grocery':
      disp = plot_partial_dependence(model, cat, [ind], line_kw={"label": cat_name,"color": color}, target=1, ax=ax, percentiles=(0,0.99))

    elif cat_name == 'Electronic Device': 
        
      disp = plot_partial_dependence(model, cat, [ind], line_kw={"label": cat_name,"color": color}, target=1, ax= disp.axes_, percentiles=(0,1))
    elif cat_name == 'Book': 
        
      disp = plot_partial_dependence(model, cat, [ind], line_kw={"label": cat_name,"color": color}, target=1, ax= disp.axes_, percentiles=(0,0.97))
    elif cat_name == 'Fashion Item': 
        
      disp = plot_partial_dependence(model, cat, [ind], line_kw={"label": cat_name,"color": color}, target=1, ax= disp.axes_, percentiles=(0,0.97))

    elif cat_name == 'Home Appliance': 
        
      disp = plot_partial_dependence(model, cat, [ind], line_kw={"label": cat_name,"color": color}, target=1, ax= disp.axes_, percentiles=(0,1))
   

  plt.yticks( 
      size = 10
      )
  plt.xticks( 
      size = 10
      )
  plt.legend(loc="upper right", prop={'size':8})    
  plt.ylim(limit)
  plt.xlabel(feature_name, size=14)
  plt.ylabel('OPRI Probability',size=14)
  plt.show()




feature_name = 'Durability'
limit = [0.3,0.6]
plot_pdp_(model, cats, colors, cat_names, feature_name, limit)

feature_name = 'Physical Appearance'
limit = [0.2,0.48]
plot_pdp_(model, cats, colors, cat_names, feature_name, limit)


i = [0.0250422854175473, 0.0250422854175473, 0.0250325583587872, 0.0253367906392457, 0.0255453269246468, 0.0250128260193761]
j = [0.34363881297993, 0.277515842019053, 0.216427778752184, 0.327871319077154, 0.401736912364754, 0.347699130850979]
feature_name = 'Performance'
limit = [0.15,0.5]
plot_pdp(model, cats, colors, cat_names, feature_name, limit, i, j, width = 0.1)




i = [0.0239991295564924,0.0239991295564924, 0.0237848773883536, 0.0237426708448887, 0.0233902815990512, 0.0231584928009628, 0.0919966632998876, 0.0919966632998876, 0.0911753633220219, 0.0910135715720734,
     0.0917076314918128]
j = [0.363468446221265, 0.314663267447788, 0.261285154859673, 0.381393370501247, 0.395410881266848, 0.363762960486754, 0.368797288751534, 0.298493660412219, 0.313536078068996, 0.402904554790456,
     0.352885301784984]
feature_name = 'Physical Appearance'
limit = [0.2,0.48]
plot_pdp(model, cats, colors, cat_names, feature_name, limit, i, j, width=0.1)


'''contour PDP'''
X_train.columns = ['Customer Service', 'Value for Money', 'Ease of Use',
       'Performance', 'Physical Appearance', 'Durability',
       'Reliability', 'Secondary Features', 'Primary Features', 'Return Management',
       'Packaging', 'Product Category', 'Review Helpfulness',
       'Review Length', 'Review Sentiment', 'Review Images']
f = X_train[X_train['Product Category'] == 3]
c = X_train[X_train['Product Category'] == 1]
b = X_train[X_train['Product Category'] == 0]
e = X_train[X_train['Product Category'] == 2]
h = X_train[X_train['Product Category'] == 4]
cats = [X_train, f, b, c, e, h
        ]

colors = ["bwr","bwr","bwr","bwr","bwr","bwr"
          ]
cat_names = ["All Categories","Food & Grocery","Book","Fashion Item","Electronic Device","Home Appliance"
             ]

font = {'fontname':'Times New Roman'}

#save all PDP results
from pandas import ExcelWriter

# from pandas.io.parsers import ExcelWriter
def save_xls(list_dfs, xls_path):
    with ExcelWriter(xls_path) as writer:
        for n, df in enumerate(list_dfs):
            df.to_excel(writer,'sheet%s' % n)
        writer.save()



def plot_pdp_contour(model, cats, colors, cat_names, feature_name1, feature_name2):
    all_cat = pd.DataFrame()
    for cat, color, cat_name in tqdm(zip(cats, colors, cat_names)):
        
        inter1 = pdp.pdp_interact(
            model=model, dataset=cat, model_features=cat.columns, features=[feature_name1, feature_name2]
            )
        column_names = inter1.pdp.columns
        values_df = inter1.pdp.rename(columns={column_names[0]:'{column_name:}_{cat:}'.format(column_name=column_names[0], cat = cat_name),
                                               column_names[1]:'{column_name:}_{cat:}'.format(column_name=column_names[1], cat = cat_name),
                                               column_names[2]:'{column_name:}_{cat:}'.format(column_name=column_names[2], cat = cat_name),
                                               }
                                      )
        all_cat = pd.concat([all_cat,values_df], axis=1)
        
        fig, axes = pdp.pdp_interact_plot(
            pdp_interact_out=inter1, feature_names=[feature_name1, feature_name2], plot_type='grid', x_quantile=False, plot_pdp=False, plot_params = {
                # plot title and subtitle
                'title': cat_name,
                'subtitle': '',
                'title_fontsize': 15,
                'subtitle_fontsize': 12,
                # color for contour line
                'contour_color':  'black',
                'font_family': 'Arial',
                # matplotlib color map for interact plot
                'cmap': 'bwr',
                # fill alpha for interact plot
                'inter_fill_alpha': 0.8,
                # fontsize for interact plot text
                'inter_fontsize': 11,})
        plt.show()
    list_interact_values.append(all_cat)

list_interact_values = []
    
#Customer Service and Packaging
plot_pdp_contour(model, cats, colors, cat_names, 'Customer Service','Packaging')  
    
#Packaging and Length of Review 
plot_pdp_contour(model, cats, colors, cat_names, 'Packaging','Review Length')    
    
#Performance (Usage)__Reliability
plot_pdp_contour(model, cats, colors, cat_names, 'Performance','Reliability')

#Return-related Operations__Packaging
plot_pdp_contour(model, cats, colors, cat_names, 'Return Management','Packaging')

#Physical Appearance__Durability
plot_pdp_contour(model, cats, colors, cat_names, 'Physical Appearance','Durability')

#Design__Return-related Operations
plot_pdp_contour(model, cats, colors, cat_names, 'Primary Features','Return Management')

#Physical Appearance__Second Features
plot_pdp_contour(model, cats, colors, cat_names, 'Physical Appearance','Secondary Features')

#Physical Appearance__Customer Sentiment
plot_pdp_contour(model, cats, colors, cat_names, 'Physical Appearance','Review Sentiment')

#Physical Appearance__Customer Sentiment
plot_pdp_contour(model, cats, colors, cat_names, 'Physical Appearance','Review Sentiment')
