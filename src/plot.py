#!/usr/bin/env python
# python-3.6

import matplotlib.pyplot as plt
import numpy as np

from helpers.read_csvs import read_csv, read_csv_sideways
from helpers.figures import Figures

if __name__ == '__main__':
    # Accuracy of all
    x, c_y, r_y = read_csv('./OUTPUT/BASE/SSE.csv', 3)
    f = Figures("Sum of Squared Error", "K Clusters", "Squared Error")
    f.start()
    f.plot_curve("cancer", x, c_y)
    f.plot_curve("reviews", x, r_y)
    f.finish()

    x, gmm, km = read_csv_sideways('./OUTPUT/BASE/reviews_adjMI.csv', 3)
    f = Figures("Reviews Mutual Information", "K Clusters", "Squared Error")
    f.start()
    f.plot_curve(km[0], x, np.isfinite(km[1]))
    f.plot_curve(gmm[0], x, np.isfinite(gmm[1]))
    f.finish()
    plt.show()

    _, x, y, label = read_csv('./OUTPUT/BASE/reviews.csv', 4)
    f = Figures("Reviews Mutual Information", "K Clusters", "Squared Error")

    x1 = [x[i] for i, l in enumerate(x) if label[i] == 1]
    y1 = [y[i] for i, l in enumerate(y) if label[i] == 1]
    x0 = [x[i] for i, l in enumerate(x) if label[i] == 0]
    y0 = [y[i] for i, l in enumerate(y) if label[i] == 0]
    f.start()
    f.plot_curve('positive', x1, y1)
    f.plot_curve('negative', x0, y0)
    f.finish()

    _, x, y, label = read_csv('./OUTPUT/BASE/cancer.csv', 4)
    f = Figures("Cancer Mutual Information", "K Clusters", "Squared Error")

    x1 = [x[i] for i, l in enumerate(x) if label[i] == 1]
    y1 = [y[i] for i, l in enumerate(y) if label[i] == 1]
    x0 = [x[i] for i, l in enumerate(x) if label[i] == 0]
    y0 = [y[i] for i, l in enumerate(y) if label[i] == 0]
    f.start()
    f.plot_curve('positive', x1, y1)
    f.plot_curve('negative', x0, y0)
    f.finish()

    plt.show()
