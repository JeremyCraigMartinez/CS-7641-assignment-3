import sys
from os.path import dirname, realpath
from time import clock
from collections import defaultdict
from multiprocessing import Process
from threading import Thread
from tempfile import mkdtemp

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE #pylint: disable=unused-import
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import Memory

from helpers.clustering import cluster_acc, myGMM, nn_arch, nn_reg, r_clusters, c_clusters
from helpers.dim_reduction import get_data

dir_path = dirname(realpath(__file__))
output_dir = sys.argv[1] if len(sys.argv) >= 2 else 'BASE'
OUT = '{}/../OUTPUT/{}'.format(dir_path, output_dir)
BASE = '{}/../OUTPUT/{}'.format(dir_path, output_dir)

r, c = get_data(BASE, "" if len(sys.argv) < 3 else sys.argv[2])
r_X, r_y = r
c_X, c_y = c

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=10)

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

    threads = []
    for k in r_clusters:
        t = Thread(target=func, args=(r_X, r_y, 'reviews', k))
        t.start()
        threads.append(t)
    for k in c_clusters:
        t = Thread(target=func, args=(c_X, c_y, 'cancer', k))
        t.start()
        threads.append(t)

    for ts in threads:
        ts.join()

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

def other(c_alg, X, y, name, clusters):
    grid = None
    if c_alg == 'km':
        grid = {'km__n_clusters':clusters, 'NN__alpha':nn_reg, 'NN__hidden_layer_sizes':nn_arch}
    else:
        grid = {'gmm__n_components':clusters, 'NN__alpha':nn_reg, 'NN__hidden_layer_sizes':nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    clust = kmeans(random_state=5) if c_alg == 'km' else myGMM(random_state=5)
    pipe = Pipeline([(c_alg, clust), ('NN', mlp)], memory=memory)
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)
    gs.fit(X, y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('{}/{} cluster {}.csv'.format(OUT, name, c_alg))

    X2D = TruncatedSVD(random_state=5).fit_transform(X)
    #X2D = TSNE(verbose=10, random_state=5).fit_transform(X)
    csv = pd.DataFrame(np.hstack((X2D, np.atleast_2d(y).T)), columns=['x', 'y', 'target'])
    csv.to_csv('{}/{}.csv'.format(OUT, name))

if __name__ == '__main__':
    f = Process(target=fit, args=())
    r_km = Process(target=other, args=('km', r_X, r_y, 'reviews', r_clusters,))
    r_gmm = Process(target=other, args=('gmm', r_X, r_y, 'reviews', r_clusters,))
    c_km = Process(target=other, args=('km', c_X, c_y, 'cancer', c_clusters,))
    c_gmm = Process(target=other, args=('gmm', c_X, c_y, 'cancer', c_clusters,))

    f.start()
    r_km.start()
    r_gmm.start()
    c_km.start()
    c_gmm.start()

    f.join()
    r_km.join()
    r_gmm.join()
    c_km.join()
    c_gmm.join()
