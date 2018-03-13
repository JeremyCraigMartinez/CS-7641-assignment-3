#!/usr/bin/env python
# python-3.6

import matplotlib.pyplot as plt

from helpers.read_csvs import read_csv
from helpers.figures import Figures

if __name__ == '__main__':
    # Accuracy of all
    x, c_y, r_y = read_csv('./OUTPUT/BASE/SSE.csv', 3)
    f = Figures("Sum of Squared Error", "K Clusters", "Squared Error")
    f.start()
    f.plot_curve("cancer", x, c_y)
    f.plot_curve("reviews", x, r_y)
    f.finish()

    #plt.plot(x, c_y, 'go')
    #plt.plot(x, r_y, 'ro')
    #plt.axis([0, 55, 5000, 80000])
    plt.show()
