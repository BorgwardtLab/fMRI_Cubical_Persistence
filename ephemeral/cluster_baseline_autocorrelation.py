import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances


X = []

for filename in sorted(glob.glob('../results/baseline_autocorrelation/brainmask/*.npz')):
    M = np.load(filename)['X']
    X.append(M.ravel())

X = np.array(X)

D = pairwise_distances(X, metric='l1')
Y = MDS(
    dissimilarity='precomputed',
    max_iter=1000,
    n_init=32,
    random_state=42,
).fit_transform(D)

df_cohorts = pd.read_csv('../data/participant_groups.csv')
groups = df_cohorts['cluster'].values

plt.scatter(Y[:, 0], Y[:, 1], c=groups, cmap='Set1')
plt.colorbar()
plt.show()
