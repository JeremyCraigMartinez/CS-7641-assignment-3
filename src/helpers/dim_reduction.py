from os.path import dirname, realpath
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from matplotlib import cm

cmap = cm.get_cmap('Spectral')

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

from helpers.clustering import nn_arch, nn_reg

def get_data(BASE):
    np.random.seed(0)
    cancer_data = pd.read_hdf('{}/cancer_datasets.hdf'.format(BASE), 'cancer')
    c_X = cancer_data.drop('Class', 1).copy().values
    c_y = cancer_data['Class'].copy().values

    reviews_data = pd.read_hdf('{}/reviews_datasets.hdf'.format(BASE), 'reviews')
    r_X = reviews_data.drop('Class', 1).copy().values
    r_y = reviews_data['Class'].copy().values

    r_X = StandardScaler().fit_transform(r_X)
    c_X = StandardScaler().fit_transform(c_X)

    return ((r_X, r_y), (c_X, c_y),)

def run_dim_alg(X, y, name, dim, dims, decomposition, OUT):
    grid = None
    if name == 'pca':
        grid = {'pca__n_components':dims, 'NN__alpha':nn_reg, 'NN__hidden_layer_sizes':nn_arch}
    elif name == 'ica':
        grid = {'ica__n_components':dims, 'NN__alpha':nn_reg, 'NN__hidden_layer_sizes':nn_arch}

    decomp = decomposition(random_state=5)
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([(name, decomp), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(X, y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('{}/{} dim red.csv'.format(OUT, name))

    decomp = decomposition(n_components=dim, random_state=10)
    X2 = decomp.fit_transform(X)
    data2 = pd.DataFrame(np.hstack((X2, np.atleast_2d(y).T)))
    cols = list(range(data2.shape[1]))
    cols[-1] = 'Class'
    data2.columns = cols
    data2.to_hdf('{}/datasets.hdf'.format(OUT, name), name, complib='blosc', complevel=9)
