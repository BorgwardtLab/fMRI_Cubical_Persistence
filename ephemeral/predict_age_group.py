import json

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

y = pd.read_csv('../data/participant_groups.csv')['cluster'].values

with open('../results/summary_statistics/brainmask.json') as f:
    data = json.load(f)

X = []

for subject in sorted(data.keys()):
    curve = data[subject]['infinity_norm_p1']
    X.append(curve)

X = np.asarray(X)
print(y)
print(X.shape)

loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    X_test = scaler.transform(X_test)
    y_pred = clf.predict(X_test)

    print(y_test, y_pred)
