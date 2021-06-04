import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from dataprocessing import *


data, y, featureinfo = load_data()

sf_data, sf_features, sf_cont = subset_community(data, featureinfo)

df_prepared, encoder, y = fit_prepare(data = sf_data, labels = y, cont = sf_cont)

X_train, X_test, y_train, y_test = train_test_split(df_prepared, y, test_size = 0.2, 
    random_state = 42)


model = RandomForestClassifier()
model.fit(X_train, y_train)
ypred = model.predict(X_test)
yposterior = model.predict_proba(X_test)
model.score(X_test, y_test)




# 
# separate the columns
# V161001 – V161523. Pre‐election interview. 
# V162001 – V162371b. Post‐election interview. 
postdata = data.filter(regex = "V162.*").drop(columns = ["V162034a"])

#162123-162125
#162128-162170

# boolean function is_political() that matches the below, apply it down the pd df
# Dem, Rep, Liberal, Conservative, President, candidate, Obama, party, Trump


# feature information data
featureinfo = pd.read_csv("Codebook/anes_timeseries_2016_varlist.csv")

# boolean function is_political() that matches the below, apply it down the pd df
# Dem, Rep, Liberal, Conservative, President, candidate, Obama, party, Trump




postdata1 = postdata.filter(regex = "^V1621[3-6][\d].*")
postdata1.mask(postdata1 < 0, np.nan).dropna(axis = 0, thresh = np.round(0.3 * postdata1.shape[1]))




# remove bad rows
pd2 = postdata1.dropna(axis=0, thresh=np.round(.1*postdata1.shape[1]))

#imput


# data is categorical --> need to either one-hot encode or 
postnp = postdata1.to_numpy()
ynp = y.to_numpy()







# one-hot encoding
encoder = OneHotEncoder()
encoder.fit(postnp[:,1:2])
X_enc = encoder.transform(postnp[:,1:2])
X_train, X_test, y_train, y_test = train_test_split(postsample, y, test_size = 0.2, 
    random_state = 42)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)
ypred = model.predict(X_test)
model.score(X_test, y_test)
text = sklearn.tree.export_text(model)

