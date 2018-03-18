#!/usr/bin/env python
# python-3.6

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

import math
def interpolate_gaps(data):
    new_data = []
    for i in range(0,len(data)):
        if not math.isnan(data[i]):
            new_data.append(data[i])
    return new_data
