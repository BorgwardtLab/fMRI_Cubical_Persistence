import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import MDS


X = []

for filename in sorted(glob.glob('../results/baseline_autocorrelation/brainmask/*.npz')):
    M = np.load(filename)['X']
    X.append(M.ravel())

X = np.array(X)

encoder = MDS(random_state=5, metric=True)
X = encoder.fit_transform(X)

df_cohorts = pd.read_csv('../data/participant_groups.csv')
groups = df_cohorts['cluster'].values

plt.scatter(X[:, 0], X[:, 1], c=groups, cmap='Set1')
plt.colorbar()
plt.show()
