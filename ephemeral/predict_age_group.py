import json

import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

y = pd.read_csv('../data/participant_groups.csv')['cluster'].values

with open('../results/summary_statistics/brainmask.json') as f:
    data = json.load(f)

X = []

for subject in sorted(data.keys()):
    curve = data[subject]['total_persistence_p2']
    X.append(curve)

X = np.asarray(X)
print(y)
print(X.shape)

loo = LeaveOneOut()
y_pred = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clf = SVC()
    clf.fit(X_train, y_train)

    X_test = scaler.transform(X_test)
    y_pred.append(*clf.predict(X_test))

C = confusion_matrix(y, y_pred)
print(C)
print(np.trace(C))
