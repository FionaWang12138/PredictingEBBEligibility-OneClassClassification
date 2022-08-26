from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

X = pd.read_csv("eval_v5.csv")
y = pd.read_csv("2022-04-14.csv")

X = X.sort_values('customer_id')
y = y.sort_values('customer_id')

X.pop('customer_id')
y.pop('customer_id')

print(X)
print(y)
model = LogisticRegression (penalty='l1', solver='liblinear', C=20).fit(X, y.values.ravel())

print(model.coef_ )

df = pd.read_csv("eval_v4.csv")
features = df.columns.values.tolist()
weight = model.coef_[0] 
pairs = np.array(list(zip(features, weight)))
print(pairs)

featuresToRemove = []
for item in pairs:
    if item[1] == '0.0':
        featuresToRemove.append(item[0])
    
print(featuresToRemove)