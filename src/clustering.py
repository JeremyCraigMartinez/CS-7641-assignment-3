from os.path import dirname, realpath
from time import clock
from collections import defaultdict
from multiprocessing import Queue
from functools import partial

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

from helpers.clustering import cluster_acc, myGMM, nn_arch, nn_reg

dir_path = dirname(realpath(__file__))
OUT = '{}/../OUTPUT/BASE'.format(dir_path)

np.random.seed(0)
cancer_data = pd.read_hdf('{}/cancer_datasets.hdf'.format(OUT), 'cancer')
c_X = cancer_data.drop('Class', 1).copy().values
c_y = cancer_data['Class'].copy().values

reviews_data = pd.read_hdf('{}/reviews_datasets.hdf'.format(OUT), 'reviews')
r_X = reviews_data.drop('Class', 1).copy().values
r_y = reviews_data['Class'].copy().values

r_X = StandardScaler().fit_transform(r_X)
c_X = StandardScaler().fit_transform(c_X)

clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]


def fit():
    st = clock()
    SSE = defaultdict(dict)
    ll = defaultdict(dict)
    acc = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))

    def func(X, y, name, it):
        km = kmeans(random_state=5)
        gmm = GMM(random_state=5)
        km.set_params(n_clusters=it)
        gmm.set_params(n_components=it)
        km.fit(X)
        gmm.fit(X)

        SSE[it][name] = km.score(X)
        ll[it][name] = gmm.score(X)
        acc[it][name]['Kmeans'] = cluster_acc(y, km.predict(X))
        acc[it][name]['GMM'] = cluster_acc(y, gmm.predict(X))
        adjMI[it][name]['Kmeans'] = ami(y, km.predict(X))
        adjMI[it][name]['GMM'] = ami(y, gmm.predict(X))
        print(it, clock()-st)

    for k in clusters:
        func(r_X, r_y, 'reviews', k)
        func(c_X, c_y, 'cancer', k)

    SSE = (-pd.DataFrame(SSE)).T
    SSE.rename(columns=lambda x: x+' SSE (left)', inplace=True)
    ll = pd.DataFrame(ll).T
    ll.rename(columns=lambda x: x+' log-likelihood', inplace=True)
    acc = pd.Panel(acc)
    adjMI = pd.Panel(adjMI)

    SSE.to_csv('{}/SSE.csv'.format(OUT))
    ll.to_csv('{}/logliklihood.csv'.format(OUT))
    acc.ix[:, :, 'reviews'].to_csv('{}/{}_acc.csv'.format(OUT, 'reviews'))
    adjMI.ix[:, :, 'reviews'].to_csv('{}/{}_adjMI.csv'.format(OUT, 'reviews'))
    acc.ix[:, :, 'cancer'].to_csv('{}/{}_acc.csv'.format(OUT, 'cancer'))
    adjMI.ix[:, :, 'cancer'].to_csv('{}/{}_adjMI.csv'.format(OUT, 'cancer'))

def other(X, y, name):
    def func(c_alg):
        grid = {'{}__n_clusters'.format(c_alg):clusters, 'NN__alpha':nn_reg, 'NN__hidden_layer_sizes':nn_arch}
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
        clust = kmeans(random_state=5) if c_alg == 'km' else myGMM(random_state=5)
        pipe = Pipeline([(c_alg, clust), ('NN', mlp)])
        gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

        gs.fit(X, y)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv('{}/{} cluster {}.csv'.format(OUT, name, c_alg))

    #func('km')
    func('gmm')

    X2D = TSNE(verbose=10, random_state=5).fit_transform(X)
    csv = pd.DataFrame(np.hstack((X2D, np.atleast_2d(y).T)), columns=['x', 'y', 'target'])
    csv.to_csv('{}/{}.csv'.format(OUT, name))

if __name__ == '__main__':
    #fit()
    other(r_X, r_y, 'reviews')
