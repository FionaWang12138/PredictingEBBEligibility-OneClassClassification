######################################################
##                                                  ##
##            TEAM 10 - ISOLATION FOREST            ##
##                                                  ##
######################################################

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import csv
import math
from sklearn.preprocessing import normalize

# importing the evaluation set
eval_set = pd.read_csv("../../data/eval_set.csv")

# importing the relevant sklearn implementation
from sklearn.ensemble import IsolationForest
data = eval_set.copy()

# data being used for isolation forest does not require one-hot encoding, but simply needs label encoding
for col in data.columns:
    if data[col].dtype == "object":
        le = preprocessing.LabelEncoder()
        data[col].fillna("None", inplace=True)
        le.fit(list(data[col].astype(str).values))
        data[col] = le.transform(list(data[col].astype(str).values))
    else:
        data[col].fillna(-999, inplace=True)

# setting the hyperparameter "contamination" to be 0.01
contamination = 0.01
model = IsolationForest(contamination=contamination, n_estimators=300)
model.fit(data.values)

# creating a new column in our dataset which will contain our predictions
data['ebb_eligible'] = model.predict(data.values)

# obtaining all of the customer IDs from the original dataset
customer_id = evaluation_set['customer_id']

# obtaining our predictions using isolation forest
ebb_eligible = data['ebb_eligible']

# replacing our negative class values from -1 to 0 to match our problem
ebb_eligible = ebb_eligible.replace(-1,0)

# combining the customer ID and associated predictions into one final output file
final = pd.concat([customer_id, ebb_eligible], axis=1)

# the predictions from this model were not used in our final analysis and submission
#final.to_csv(r'2022-04-17.csv', index=False,header=True)