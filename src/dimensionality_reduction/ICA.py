from os.path import dirname, realpath
import sys

import pandas as pd
from sklearn.decomposition import FastICA
import numpy as np

np.random.seed(0)

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.dim_reduction import run_dim_alg, get_data

r_dims = c_dims = [i for i in range(5, 62, 8)]

OUT = '{}/../../OUTPUT/ICA'.format(dir_path)
BASE = '{}/../../OUTPUT/BASE'.format(dir_path)

r, c = get_data(BASE)
r_X, r_y = r
c_X, c_y = c

ica = FastICA(random_state=5)
kurt = {}
for dim in r_dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(r_X)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv('{}/reviews kurtosis.csv'.format(OUT))

ica = FastICA(random_state=5)
kurt = {}
for dim in c_dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(c_X)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv('{}/cancer kurtosis.csv'.format(OUT))

def main():
    decomp1 = FastICA(random_state=10)
    decomp2 = FastICA(random_state=10)
    for i, val in enumerate(r_dims): # [0.6, 0.7, 0.8, 0.9]:
        decomp1.set_params(n_components=r_dims[i])
        decomp2.set_params(n_components=c_dims[i])
        run_dim_alg(r_X, r_y, 'ica', 'reviews', decomp1, r_dims[i], OUT)
        run_dim_alg(c_X, c_y, 'ica', 'cancer', decomp2, c_dims[i], OUT)

if __name__ == '__main__':
    #main()
    print('do nothing')
