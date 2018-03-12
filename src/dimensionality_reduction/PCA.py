from os.path import dirname, realpath
import sys

import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import cm

cmap = cm.get_cmap('Spectral')

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.dim_reduction import run_dim_alg, get_data

#sys.path.insert(0, '/Users/jeremy.martinez/georgia-tech-code/ass3/jontay/src')
OUT = '{}/../../OUTPUT/PCA'.format(dir_path)
BASE = '{}/../../OUTPUT/BASE'.format(dir_path)

r, c = get_data(BASE)
r_X, r_y = r
c_X, c_y = c

clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
dims = [2, 5, 10, 15, 20, 25, 30, 35, 40]

pca = PCA(random_state=5)
pca.fit(r_X)
series = pd.Series(data=pca.explained_variance_, index=range(1, 101))
series.to_csv('{}/reviews screen.csv'.format(OUT))

pca = PCA(random_state=5)
pca.fit(c_X)
series = pd.Series(data=pca.explained_variance_, index=range(1, 47))
series.to_csv('{}/cancer screen.csv'.format(OUT))

#%% Data for 2

run_dim_alg(r_X, r_y, 'pca', 5, dims, PCA, OUT)
run_dim_alg(c_X, c_y, 'pca', 30, dims, PCA, OUT)
