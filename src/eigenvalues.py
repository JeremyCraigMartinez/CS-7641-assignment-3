from os.path import dirname, realpath
import sys

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from helpers.figures import Figures
from helpers.dim_reduction import get_data

np.random.seed(0)

cmap = cm.get_cmap('Spectral')

dir_path = dirname(realpath(__file__))
sys.path.insert(0, '{}/..'.format(dir_path))

#sys.path.insert(0, '/Users/jeremy.martinez/georgia-tech-code/ass3/jontay/src')
OUT = '{}/../OUTPUT/PCA'.format(dir_path)
BASE = '{}/../OUTPUT/BASE'.format(dir_path)

r, c = get_data(BASE)
r_X, _ = r
c_X, _ = c

_plot_colors = ('r', 'g', 'b', 'c', 'darkorange', 'black')

def main():
    rpca = PCA(random_state=5)
    rpca.fit(r_X)
    r_eigen = rpca.explained_variance_ratio_

    cpca = PCA(random_state=5)
    cpca.fit(c_X)
    c_eigen = cpca.explained_variance_ratio_
    nanarr = np.empty((len(r_eigen) - len(c_eigen)))
    nanarr = 0
    nanarr = np.array(nanarr)
    c_eigen = np.append(c_eigen, np.zeros(54) + np.nan)

    f = Figures("Eigenvalue Distribution", "N Components", "Eigenvalue")
    f.start()
    f.plot_curve("cancer", range(len(r_eigen)), c_eigen, plot_colors=_plot_colors)
    f.plot_curve("review", range(len(r_eigen)), r_eigen, plot_colors=_plot_colors)
    f.finish()
    plt.show()

if __name__ == '__main__':
    main()
