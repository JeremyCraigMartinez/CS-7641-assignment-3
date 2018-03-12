from os.path import dirname, realpath
import sys
from functools import partial

#%% Imports
import pandas as pd
from sklearn.decomposition import FastICA

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.clustering import clusters, dims
from helpers.dim_reduction import run_dim_alg, get_data

OUT = '{}/../../OUTPUT/ICA'.format(dir_path)
BASE = '{}/../../OUTPUT/BASE'.format(dir_path)

r, c = get_data(BASE)
r_X, r_y = r
c_X, c_y = c

ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(r_X)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv('{}/reviews screen.csv'.format(OUT))

ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(c_X)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv('{}/cancer screen.csv'.format(OUT))

init_decomp = partial(FastICA, random_state=10)
decomp1 = partial(FastICA, n_components=5, random_state=10)
decomp2 = partial(FastICA, n_components=30, random_state=10)
run_dim_alg(r_X, r_y, 'pca', dims, (init_decomp, decomp1), OUT)
run_dim_alg(c_X, c_y, 'pca', dims, (init_decomp, decomp2), OUT)
