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

def read_csv_sideways(path_name, rows=2):
    data = pd.read_csv(path_name, sep=",")
    csv = [np.array(data.columns[1:]).astype(np.double)]
    for i in range(rows):
        csv.append((data.values[i][0], np.array(data.values[i][1:]).astype(np.double)))
    return csv

def interpolate_gaps(data):
    new_data = []
    for _, val in enumerate(data):
        if not math.isnan(val):
            new_data.append(val)
    return new_data
