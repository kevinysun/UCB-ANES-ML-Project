import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from dataprocessing import *

def fit_prepare(data, labels, drop_thresh = 0.4, cont = []):
    """
    Take subsetted ANES features and labels and process them for random forest training
    Includes: missing value imputation and one-hot encoding. One-hot encoding is only for
    categorical variables - continuous variables can be specified and ignored

    ARGUMENTS
    df: pandas dataframe
    drop_thresh: proportion of missing values threshold, if over this we drop the row
    cont: list of continuous variables to not use one-hot encoding with
    y: labels
    outputs: encoded ndarray to fit on


    ISSUE: need to somehow be able to identify columns
    """
    df = data.copy()
    pshape = df.shape
    df.insert(0, "y", labels.values)
    df = (df.mask(df < 0, np.nan)
                .dropna(axis = 0, thresh = np.round(drop_thresh * pshape[1]))
                )

    y_out = df['y']
    
    df = df.drop(columns = ['y'])
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)  
    
    #knn imputer
    imputer = KNNImputer(n_neighbors=10, weights='uniform')


    df_imputed = pd.DataFrame(imputer.fit_transform(df),
                                    columns = df.columns)
    #rescale

    df_imputed = scaler.inverse_transform(df_imputed)
    df_imputed = pd.DataFrame(df_imputed, columns = df.columns)
    # knn imputer gives decimals, so we round to integers
    # ISSUE: continuous columns?
    # Approach: we will pass a list of column names to ignore
    # create df_cat which drops non-categorical
    if not cont:
        df_cat = df_imputed.copy()
    else:
        df_cat = df_imputed.drop(cont, axis=1) 

    #cat_vars = df_cat.columns
    df_cat = np.round(df_cat).astype(np.int32)

    if cont:
        df_cont = df_imputed[cont].to_numpy()
    
    #one-hot encoding
    encoder = OneHotEncoder()
    encoder.fit(df_cat)
    df_enc = encoder.transform(df_cat).A

    if not cont:
        df_prepared = df_enc
    else:
        df_prepared = np.hstack((df_cont, df_enc))

    return df_prepared, encoder, y_out

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier 
import matplotlib.pyplot as plt

# load data
data, y, featureinfo = load_data()


subset_fns = [subset_community, subset_gender, subset_hcsci, subset_intl, subset_natlism,
                subset_other, subset_pers, subset_polit_sys, subset_race, 
                subset_social_financial]

topics = ['community', 'gender', 'healthcare/science', 'intl relations', 'nationalism',
            'miscellaneous', 'personality/values', 'political system', 'race', 'social/financial']

bestparamslistrf = []
scoresrf = []
# tune each forest
for fn in subset_fns:
    sub_data, sub_features, sub_cont = fn(data, featureinfo)
    X_prepared, encoder, y_prep = fit_prepare(data = sub_data, labels = y, cont = sub_cont)

    model = RandomForestClassifier()
    params = {'criterion' : ('gini', 'entropy'), 
               'max_depth': [1, 3, 10, 20]}
    tuner = GridSearchCV(estimator = model, param_grid = params)
    tuner.fit(X_prepared, y_prep)

    bestparamslistrf.append(tuner.best_params_)
    scoresrf.append(tuner.best_score_)

# pandas df for best parameters
bestparamsrf = pd.DataFrame(bestparamslistrf)
bestparamsrf['Subject'] = topics
bestparamsrf['Score'] = scoresrf

bestparamsrf.columns = ['Criterion', 'Max Depth', 'Subject', 'Score']
bestparamsrf = bestparamsrf[['Subject', 'Score', 'Criterion', 'Max Depth']]
summaryrf = bestparamsrf

# plot
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(topics,scoresrf)
ax.set_ylabel('Score')
ax.set_xlabel('Variable Group')
ax.set_title('Random Forest Scores')
ax.set_xticklabels(labels = topics, rotation = 80)
ax.set_ylim(bottom = 0.5, top = 1)
plt.show()
fig.savefig("randomforestgraph.png", bbox_inches='tight')