# code found at http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html

import numpy as np
import itertools
from os.path import dirname, realpath
import sys

from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture

from helpers.dim_reduction import get_data

print(__doc__)

dir_path = dirname(realpath(__file__))
output_dir = sys.argv[1] if len(sys.argv) >= 2 else 'BASE'
OUT = '{}/../OUTPUT/{}'.format(dir_path, output_dir)
BASE = '{}/../OUTPUT/BASE'.format(dir_path)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.
_r, _c = get_data(BASE)
r_X, r_y = _r
c_X, c_y = _c

r_components = [8, 13, 21, 34, 55, 89, 104, 119, 134, 159]
c_components = [8, 10, 14, 18, 25, 35, 45, 55, 65, 75]
n_components_range = r_components

def main(X, _):
    lowest_bic = np.infty
    bic = []
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    print(clf)
    bars = []

    # Plot the BIC scores
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + 1.7 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=1.7, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
        .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    #Y_ = clf.predict(X)
    #for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
    #                                           color_iter)):
    #    print(cov)
    #    v, w = linalg.eigh(cov)
    #    if not np.any(Y_ == i):
    #        continue
    #    plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
    #    # Plot an ellipse to show the Gaussian component
    #    angle = np.arctan2(w[0][1], w[0][0])
    #    angle = 180. * angle / np.pi  # convert to degrees
    #    v = 2. * np.sqrt(2.) * np.sqrt(v)
    #    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    #    ell.set_clip_box(splot.bbox)
    #    ell.set_alpha(.5)
    #    splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model, 2 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()

color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
                              'darkorange'])

def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

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

def epsilloids(X, n):
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=n, covariance_type='full').fit(X)
    plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
                 'Gaussian Mixture')

    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=n,
                                            covariance_type='full').fit(X)
    plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
                 'Bayesian Gaussian Mixture with a Dirichlet process prior')

    plt.show()

if __name__ == '__main__':
    #main(r_X, r_y)
    #main(c_X, c_y)
    epsilloids(r_X, 55)
