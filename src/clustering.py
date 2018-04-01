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

args = sys.argv[2:]
if len(args) >= 3:
    args = args if args[1] != '--delim' else args[2].split(',')
print(args)
dir_path = dirname(realpath(__file__))
input_dir = args[0] if len(args) >= 1 else 'BASE'
output_dir = '{}/{}'.format(input_dir, args[1][:-1]) if len(args) >= 2 else input_dir
print(input_dir, output_dir)
OUT = '{}/../OUTPUT/{}'.format(dir_path, output_dir)
BASE = '{}/../OUTPUT/{}'.format(dir_path, input_dir)

if '-r' in sys.argv:
    r = get_data(BASE, "" if len(args) < 2 else args[1], 'r')
    r_X, r_y = r
elif '-c' in sys.argv:
    c = get_data(BASE, "" if len(args) < 2 else args[1], 'c')
    c_X, c_y = c
else:
    r, c = get_data(BASE, "" if len(args) < 2 else args[1])
    r_X, r_y = r
    c_X, c_y = c

cachedir = mkdtemp()
memory = Memory(cachedir=cachedir, verbose=10)

def file_it(name, alg, X, y, y_pred, it=None):
    tmp = []
    for i, _ in enumerate(X):
        tmp.append(np.append(X[i], y_pred[i]))
    X2 = np.array(tmp)
    print('NUM CLUSTERS: %s' % it)
    data2 = pd.DataFrame(np.hstack((X2, np.atleast_2d(y).T)))
    cols = list(range(data2.shape[1]))
    cols[-1] = 'Class'
    data2.columns = cols
    data2.to_hdf('{}/datasets-w-cluster/{}-{}-datasets.{}.hdf'.format(BASE, it, args[1][:-1], alg), name, complib='blosc', complevel=9)

def fit(ignore=None):
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

        file_it(name, 'km', X, y, km.predict(X), it=it)
        file_it(name, 'gmm', X, y, gmm.predict(X), it=it)

        SSE[it][name] = km.score(X)
        ll[it][name] = gmm.score(X)
        acc[it][name]['Kmeans'] = cluster_acc(y, km.predict(X))
        acc[it][name]['GMM'] = cluster_acc(y, gmm.predict(X))
        adjMI[it][name]['Kmeans'] = ami(y, km.predict(X))
        adjMI[it][name]['GMM'] = ami(y, gmm.predict(X))
        print(it, clock()-st)

    threads = []
    if ignore != 'reviews':
        for k in r_clusters:
            t = Thread(target=func, args=(r_X, r_y, 'reviews', k))
            t.start()
            threads.append(t)
    if ignore != 'cancer':
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
    if ignore != 'reviews':
        acc.ix[:, :, 'reviews'].to_csv('{}/{}_acc.csv'.format(OUT, 'reviews'))
        adjMI.ix[:, :, 'reviews'].to_csv('{}/{}_adjMI.csv'.format(OUT, 'reviews'))
    if ignore != 'cancer':
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
    _ignore = None
    if '-r' in sys.argv:
        _ignore = 'cancer'
    elif '-c' in sys.argv:
        _ignore = 'reviews'
    else:
        print('run "-r" or "-c" to isolate reivew or cancer data')

    processes = []
    processes.append(Process(target=fit, args=(_ignore,)))
    #if _ignore != 'reviews':
    #    processes.append(Process(target=other, args=('km', r_X, r_y, 'reviews', r_clusters,)))
    #    processes.append(Process(target=other, args=('gmm', r_X, r_y, 'reviews', r_clusters,)))
    #if _ignore != 'cancer':
    #    processes.append(Process(target=other, args=('km', c_X, c_y, 'cancer', c_clusters,)))
    #    processes.append(Process(target=other, args=('gmm', c_X, c_y, 'cancer', c_clusters,)))

    for p in processes:
        p.start()

    for p in processes:
        p.join()
