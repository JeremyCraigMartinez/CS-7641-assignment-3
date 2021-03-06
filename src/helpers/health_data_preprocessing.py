from os.path import dirname, realpath

from sklearn import preprocessing
import pandas as pd
import numpy as np

dir_path = dirname(realpath(__file__))

# categorical data columns containing missing values
categorical_data = [
    'Smokes',
    'Hormonal Contraceptives',
    'STDs',
]

# continuous data columns containing missing values
continuous_data = [
    'Number of sexual partners',
    'First sexual intercourse',
    'Num of pregnancies',
    'Smokes (years)',
    'Smokes (packs/year)',
    'Hormonal Contraceptives (years)',
    'STDs (number)',
    'STDs:condylomatosis',
    'STDs:cervical condylomatosis',
    'STDs:vaginal condylomatosis',
    'STDs:vulvo-perineal condylomatosis',
    'STDs:syphilis',
    'STDs:pelvic inflammatory disease',
    'STDs:genital herpes',
    'STDs:molluscum contagiosum',
    'STDs:AIDS',
    'STDs:HIV',
    'STDs:Hepatitis B',
    'STDs:HPV',
    'STDs: Time since first diagnosis',
    'STDs: Time since last diagnosis',
]

# Treat this data differently because IUD (years)
# is dependent on IUD and so these rows need to carefully
# be filled together (or just not at all... so that's what we'll do)
special_data = [
    'IUD',
    'IUD (years)',
]

def fill_missing_data(data, columns, func):
    for entry in columns:
        data[entry] = func(data, entry)

    return data

def clean(dataset):
    data = dataset.replace('?', np.nan)

    data = data.convert_objects(convert_numeric=True)

    # fill missing data
    data = fill_missing_data(data, continuous_data, lambda d, k: d[k].fillna(d[k].median()))
    data = fill_missing_data(data, categorical_data, lambda d, k: d[k].fillna(1))
    data = fill_missing_data(data, special_data, lambda d, k: d[k].fillna(0))

    # for categorical data
    all_cat_cols = [
        'Smokes',
        'Hormonal Contraceptives',
        'IUD',
        'STDs', 'Dx:Cancer',
        'Dx:CIN',
        'Dx:HPV',
        'Dx',
        'Hinselmann',
        'Citology',
        'Schiller'
    ]
    data = pd.get_dummies(data=data, columns=all_cat_cols)

    return data

def feature_scale(data):
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_data = minmax_scale.fit_transform(data)
    return scaled_data

def get_X_Y(dataset):
    X_cols = list(dataset.columns.values)
    y_cols = [X_cols.pop(X_cols.index('Biopsy'))]

    np.random.seed(42) #pylint: disable=no-member
    data_shuffle = dataset.iloc[np.random.permutation(len(dataset))] #pylint: disable=no-member

    # split data into features and labels
    y = data_shuffle[y_cols]
    X = data_shuffle[X_cols]

    # scale features so no one features are weighed unevenly
    X = feature_scale(X)

    y = np.array(y)

    return X, y

def get_train_test_set():
    dataset = pd.read_csv('{}/../../data/kag_risk_factors_cervical_cancer.csv'.format(dir_path))
    cleaned_data = clean(dataset)

    X, y = get_X_Y(cleaned_data)
    return X, y
