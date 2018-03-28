# code found at http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html

import itertools
from os.path import dirname, realpath
import sys

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

from helpers.dim_reduction import get_data

print(__doc__)

dir_path = dirname(realpath(__file__))
output_dir = sys.argv[1] if len(sys.argv) >= 2 else 'BASE'
OUTPUT = '{}/../OUTPUT'.format(dir_path)
OUT = '{}/{}'.format(OUTPUT, output_dir)
BASE = '{}/BASE'.format(OUTPUT)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
_r, _c = get_data(BASE)
r_X, r_y = _r
c_X, c_y = _c

r_components = [8, 13, 21, 34, 55, 89, 104, 119, 134, 159]
c_components = [8, 10, 14, 18, 25, 35, 45, 55, 65, 75]
n_components_range = None
def find_best(X, bic): #pylint: disable
    cv_types = ['spherical', 'tied', 'diag', 'full']
    lowest_bic = np.infty
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            thebic = gmm.bic(X)
            bic.append(thebic)

            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
    return lowest_bic

def comparePCA(p, ds):
    global BASE
    bests = []
    params = p
    BASE = '{}/PCA'.format(OUTPUT)
    for param in params:
        _r, _c = get_data(BASE, param)
        r, _ = _r
        find_best(r, bests)

    bests = np.array(bests)
    plot_gmms(bests, ds)

def compare(dim_red_alg, p1, p2):
    global BASE
    r_bests = []
    c_bests = []
    params = p1
    BASE = '{}/{}'.format(OUTPUT, dim_red_alg)
    for param in params:
        _r, _c = get_data(BASE, param)
        c, _ = _c
        find_best(c, c_bests)

    params = p2
    BASE = '{}/{}'.format(OUTPUT, dim_red_alg)
    for param in params:
        _r, _c = get_data(BASE, param)
        r, _ = _r
        find_best(r, r_bests)

    r_bests = np.array(r_bests)
    c_bests = np.array(c_bests)
    print(r_bests)

    plot_gmms(r_bests, "Reviews")
    plot_gmms(c_bests, "Cancer")

def plot_gmms(bic, ds=""):
    cv_types = ['spherical', 'tied', 'diag', 'full']
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    bars = []

    # Plot the BIC scores
    spl = plt.subplot(2, 1, 1)
    for j, (_, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 1.7 * (j - 2)
        bars.append(plt.bar(xpos, bic[j * len(n_components_range):
                                      (j + 1) * len(n_components_range)],
                            width=1.7, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('%s BIC score per model' % ds)
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2) # pylint: disable=W0612

    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model, 2 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()

def plot_results(X, Y_, means, covariances, index, title):
    color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold', 'darkorange'])

    splot = plt.subplot(2, 1, 1 + index)
    for j, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == j):
            continue
        plt.scatter(X[Y_ == j, 0], X[Y_ == j, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-9., 5.)
    plt.ylim(-3., 6.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)

def epsilloids(X, n, covariance_type='full'):
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=n, covariance_type=covariance_type).fit(X)
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
                 'Gaussian Mixture')

    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=n,
                                            covariance_type=covariance_type).fit(X)
    plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
                 'Bayesian Gaussian Mixture with a Dirichlet process prior')

    plt.show()

if __name__ == '__main__':
    print('to run: python src/bic.py PCA')
    if '--epsilloids' in sys.argv:
        epsilloids(r_X, 55)
    else:
        if 'PCA' in sys.argv:
            r_components = [8, 13, 21, 34, 55, 89, 104, 119, 134, 159]
            c_components = [8, 10, 14, 18, 25, 35, 45, 55, 65, 75]
            n_components_range = r_components
            datasets = ['0.6-', '0.7-', '0.8-', '0.9-']
            for i in datasets:
                comparePCA([i], 'Reviews')
            n_components_range = c_components
            for i in datasets:
                comparePCA([i], 'Cancer')
            plt.show()
        if 'ICA' in sys.argv:
            n_components_range = [i for i in range(5, 62, 8)]
            # ['37-']#, '0.7-', '0.8-', '0.9-', '0.95-']
            # ['45-']#, '0.7-', '0.8-', '0.9-', '0.95-']
            compare('ICA', ['37-'], ['45-'])
        if 'RP' in sys.argv:
            n_components_range = [i for i in range(5, 62, 8)]
            r_dims = c_dims = ['%s-' % i for i in range(16, 80, 7)]
            for i in r_dims:
                compare('RP', [i], [i])
            #compare('RP', [], ['72-'])
            #compare('RP', [], ['65-'])
