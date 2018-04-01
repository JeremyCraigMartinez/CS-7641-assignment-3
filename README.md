# How to run locally

For a better view, see the [Github README](https://github.com/JeremyCraigMartinez/CS-7641-assignment-3/README.md)

Reports:
1) jmartinez91-analysis.pdf - report

## Code Files
```
src
├── bic.py - plot bic's
├── clustering.py - where the magic happens. Running KMeans and EM clustering - generating csvs for accuracy, MI, SSE, loglikelihood, etc. Some saved of csvs are irrelevant*
├── dimensionality_reduction - all scripts for dimensionality reduction
│   ├── ICA.py
│   ├── PCA.py
│   ├── RP.py
│   ├── SVD.py
│   └── SVD_extra.py - extra file - not used in analysis but used it as a reference for SVD (from sklearn docs)
├── eigenvalues.py - generate and plot eigenvalues
├── helpers
│   ├── clustering.py - helpers for clustering.py
│   ├── constants.py - arrays that don't change
│   ├── dim_reduction.py - helper file with function for getting the data and saving transformed data to HDF file
│   ├── figures.py - helper for plotting figures
│   ├── health_data_preprocessing.py - preprocessing file for cancer data (nearly same as Assignments 1 & 2)
│   ├── read_csvs.py - helper file (for figures.py) for reading data from csv files
│   ├── reviews_preprocessing.py - preprocessing file for review data (nearly same as Assignments 1 & 2)
│   └── scoring.py - small helper for finding accuracy of prediction
├── nn_dim_red_data.py - gather all dimension reduced data for comparison (finding max accuracy) and plotting
├── nn_dim_with_cluster_data.py - gather all km/em cluster data for comparison (finding max accuracy) and plotting
├── parse.py - process csv data files in HDF files
├── plot.py - script to plot (nearly) all of the graphs
└── silhouette.py - plot silhouette's
```

Note: Irrelevant csv files (mentioned in clustering.py) saved off were files I thought would be handy but I didn't end up using. Please ignore that code (sorry for the clutter!)

## output data files

```
OUTPUT
├── BASE
├── ICA
├── RP
├── SVD
├── PCA (e.g... same for all four above)
│   ├── 0.6 (dimensionality)
│   │   ├── csv's for accuracy, SSE, loglikelihood, etc.
│   ├── 0.6-datasets.hdf

```

## How to run locally

#### Install dependencies

```
# Using python 3.6
# don't muddy up your dev environment, use venv
python -m venv .venv
source .venv/bin/activate
# install dependencies in your virtual environment
pip install -r requirements.txt
```

#### Running the code

```
# create directories for data and parse initial data
./scripts/create_dirs
python src/parse.py
```

```
# run clustering
python src/clustering.py --delim BASE

# run dimensionality reduction
python src/dimensionality_reduction/PCA.py
python src/dimensionality_reduction/ICA.py
python src/dimensionality_reduction/SVD.py
python src/dimensionality_reduction/RP.py

# run rest of clustering (on reduced data)
python src/clustering.py --delim PCA 0.6- -r &
python src/clustering.py --delim PCA 0.7- -r &
python src/clustering.py --delim PCA 0.8- -r &
python src/clustering.py --delim PCA 0.9- -r

python src/clustering.py --delim ICA 13- -r &
python src/clustering.py --delim ICA 21- -r &
python src/clustering.py --delim ICA 29- -r &
python src/clustering.py --delim ICA 37- -r &
python src/clustering.py --delim ICA 45- -r &
python src/clustering.py --delim ICA 5-  -r &
python src/clustering.py --delim ICA 53- -r &
python src/clustering.py --delim ICA 61- -r

python src/clustering.py --delim RP 16- -r &
python src/clustering.py --delim RP 23- -r &
python src/clustering.py --delim RP 30- -r &
python src/clustering.py --delim RP 37- -r &
python src/clustering.py --delim RP 44- -r
python src/clustering.py --delim RP 51- -r &
python src/clustering.py --delim RP 58- -r &
python src/clustering.py --delim RP 65- -r &
python src/clustering.py --delim RP 72- -r &
python src/clustering.py --delim RP 79- -r

python src/clustering.py --delim SVD 16- -r &
python src/clustering.py --delim SVD 23- -r &
python src/clustering.py --delim SVD 30- -r &
python src/clustering.py --delim SVD 37- -r &
python src/clustering.py --delim SVD 44- -r &
python src/clustering.py --delim SVD 51- -r
python src/clustering.py --delim SVD 58- -r &
python src/clustering.py --delim SVD 65- -r &
python src/clustering.py --delim SVD 72- -r &
python src/clustering.py --delim SVD 79- -r &
python src/clustering.py --delim SVD 8- -r
```

```
# this will take a long time...
python src/plot.py
```

```
# plot a specific silhouette graph
# these take a while
python silhouette PCA
python silhouette ICA
python silhouette RP
python silhouette SVD
```

```
# plot a specific BIC graph
# these take a while
python src/bic.py PCA
etc...
```

```
# plot Neural Networks
python src/nn_dim_red_data.py
# with clusters as attributes
python src/nn_dim_with_cluster_data.py
```
