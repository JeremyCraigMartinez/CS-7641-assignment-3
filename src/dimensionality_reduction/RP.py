from os.path import dirname, realpath
from collections import defaultdict
from itertools import product
import sys
from functools import partial

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from matplotlib import cm

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.clustering import pairwiseDistCorr, nn_reg, nn_arch, reconstructionError, clusters, dims
from helpers.dim_reduction import run_dim_alg, get_data

OUT = '{}/../../OUTPUT/RP'.format(dir_path)
BASE = '{}/../../OUTPUT/BASE'.format(dir_path)

r, c = get_data(BASE)
r_X, r_y = r
c_X, c_y = c

tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(r_X), r_X)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('{}/review scree1.csv'.format(OUT))


tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(c_X), c_X)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('{}/cancer scree1.csv'.format(OUT))


tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(r_X)
    tmp[dim][i] = reconstructionError(rp, r_X)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('{}/review scree2.csv'.format(OUT))


tmp = defaultdict(dict)
for i, dim in product(range(10), dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(c_X)
    tmp[dim][i] = reconstructionError(rp, c_X)
tmp =pd.DataFrame(tmp).T
tmp.to_csv('{}/cancer scree2.csv'.format(OUT))

init_decomp = partial(SparseRandomProjection, random_state=5)
decomp1 = partial(SparseRandomProjection, n_components=5, random_state=5)
decomp2 = partial(SparseRandomProjection, n_components=30, random_state=5)
run_dim_alg(r_X, r_y, 'rp', dims, (init_decomp, decomp1), OUT)
run_dim_alg(c_X, c_y, 'rp', dims, (init_decomp, decomp2), OUT)
