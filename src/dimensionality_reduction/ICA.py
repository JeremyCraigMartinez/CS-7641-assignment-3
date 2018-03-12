from os.path import dirname, realpath
import sys

#%% Imports
import pandas as pd
from sklearn.decomposition import FastICA

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.dim_reduction import run_dim_alg, get_data

#sys.path.insert(0, '/Users/jeremy.martinez/georgia-tech-code/ass3/jontay/src')
OUT = '{}/../../OUTPUT/ICA'.format(dir_path)
BASE = '{}/../../OUTPUT/BASE'.format(dir_path)

r, c = get_data(BASE)
r_X, r_y = r
c_X, c_y = c

clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]
dims = [2, 5, 10, 15, 20, 25, 30, 35, 40]
#raise
#%% data for 1

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

run_dim_alg(r_X, r_y, 'ica', 5, dims, FastICA, OUT)
run_dim_alg(c_X, c_y, 'ica', 30, dims, FastICA, OUT)
