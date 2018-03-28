#!/usr/bin/env python
# python-3.6

import math
import pandas as pd
import numpy as np

def read_csv(path_name, columns):
    data = pd.read_csv(path_name, sep=",")
    size = len(data.values)
    cols = []
    for c in range(columns):
        cols.append(data.values[0:size, c])
    return cols

def read_csv_sideways(path_name):
    data = pd.read_csv(path_name, sep=",")
    return (
        np.array(data.columns[1:]).astype(np.double),
        (data.values[0][0], np.array(data.values[0][1:]).astype(np.double)),
        (data.values[1][0], np.array(data.values[1][1:]).astype(np.double)),
    )

def interpolate_gaps(data):
    new_data = []
    for _, val in enumerate(data):
        if not math.isnan(val):
            new_data.append(val)
    return new_data
