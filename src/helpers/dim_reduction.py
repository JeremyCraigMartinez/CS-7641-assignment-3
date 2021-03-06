from os.path import dirname, realpath
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from matplotlib import cm

cmap = cm.get_cmap('Spectral')

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.clustering import nn_arch, nn_reg

def get_data_split(BASE, prefix="", ds=None, suffix="datasets.hdf"):
    data = get_data(BASE, prefix=prefix, ds=ds, suffix=suffix)
    if ds is None:
        r, c = data
        r = train_test_split(r[0], r[1])
        c = train_test_split(c[0], c[1])
        return r, c
    return train_test_split(data[0], data[1])

def get_data(BASE, prefix="", ds=None, suffix="datasets.hdf"):
    np.random.seed(0)
    print('{}/{}{}'.format(BASE, prefix, suffix))
    def cancer():
        cancer_data = pd.read_hdf('{}/{}{}'.format(BASE, prefix, suffix), 'cancer')
        c_X = cancer_data.drop('Class', 1).copy().values
        c_y = cancer_data['Class'].copy().values
        c_X = StandardScaler().fit_transform(c_X)
        return (c_X, c_y)

    def reviews():
        reviews_data = pd.read_hdf('{}/{}{}'.format(BASE, prefix, suffix), 'reviews')
        r_X = reviews_data.drop('Class', 1).copy().values
        r_y = reviews_data['Class'].copy().values
        r_X = StandardScaler().fit_transform(r_X)
        return (r_X, r_y)

    if ds == 'c':
        return cancer()
    if ds == 'r':
        return reviews()

    return (reviews(), cancer(),)

def run_dim_alg(X, y, dname, decomp, p, OUT):
    X2 = decomp.fit_transform(X)
    data2 = pd.DataFrame(np.hstack((X2, np.atleast_2d(y).T)))
    cols = list(range(data2.shape[1]))
    cols[-1] = 'Class'
    data2.columns = cols
    data2.to_hdf('{}/{}-datasets.hdf'.format(OUT, p), dname, complib='blosc', complevel=9)

# just for testing locally, ignore below blocks
def run_nn_with_eigen_weight_vector(X, y, name, dname, dims, decomposition, OUT, filt=False):
    grid = None
    if name == 'pca':
        grid = {'pca__n_components':dims, 'NN__alpha':nn_reg, 'NN__hidden_layer_sizes':nn_arch}
    elif name == 'ica':
        grid = {'ica__n_components':dims, 'NN__alpha':nn_reg, 'NN__hidden_layer_sizes':nn_arch}
    elif name == 'rp':
        grid = {'rp__n_components':dims, 'NN__alpha':nn_reg, 'NN__hidden_layer_sizes':nn_arch}
    elif name == 'rf':
        grid = {'filter__n':dims, 'NN__alpha':nn_reg, 'NN__hidden_layer_sizes':nn_arch}

    decomp = decomposition()
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    arg = (name, decomp) if filt is False else ('filter', decomp)
    pipe = Pipeline([arg, ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(X, y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('{}/{}_{}_dim_red.csv'.format(OUT, dname, name))
