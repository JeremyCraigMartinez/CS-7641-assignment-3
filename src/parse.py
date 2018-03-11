import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
import os
import sklearn.model_selection as ms

OUT = './OUTPUT/BASE/'
cancer = pd.read_csv('./data/kag_risk_factors_cervical_cancer.csv',header=None,sep=',')
reviews = pd.read_csv('./data/Restaurant_Reviews.tsv.data',header=None,sep='\t')

c_train_X, c_test_X, c_train_y, c_test_y = ms.train_test_split(madX, madY, test_size=0.2, random_state=0, stratify=madY)

madX = pd.DataFrame(madelon_trgX)
madY = pd.DataFrame(madelon_trgY)
madY.columns = ['Class']

madX2 = pd.DataFrame(madelon_tstX)
madY2 = pd.DataFrame(madelon_tstY)
madY2.columns = ['Class']

mad1 = pd.concat([madX,madY],1)
mad1 = mad1.dropna(axis=1,how='all')
mad1.to_hdf(OUT+'datasets.hdf','madelon',complib='blosc',complevel=9)

mad2 = pd.concat([madX2,madY2],1)
mad2 = mad2.dropna(axis=1,how='all')
mad2.to_hdf(OUT+'datasets.hdf','madelon_test',complib='blosc',complevel=9)

digits = load_digits(return_X_y=True)
digitsX,digitsY = digits

digits = np.hstack((digitsX, np.atleast_2d(digitsY).T))
digits = pd.DataFrame(digits)
cols = list(range(digits.shape[1]))
cols[-1] = 'Class'
digits.columns = cols
digits.to_hdf(OUT+'datasets.hdf','digits',complib='blosc',complevel=9)

