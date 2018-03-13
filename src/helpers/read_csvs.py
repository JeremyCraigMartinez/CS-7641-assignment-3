#!/usr/bin/env python
# python-3.6

import pandas as pd

def read_csv(path_name, columns):
    data = pd.read_csv(path_name, sep=",")
    size = len(data.values)
    cols = []
    for c in range(columns):
        cols.append(data.values[0:size, c])
    return cols
