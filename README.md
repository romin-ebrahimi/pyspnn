# Python SPNN
Scale Invariant Probabilistic Neural Networks for Python.

- [SPNN Paper](https://repositories.lib.utexas.edu/items/b1818ab2-c2a8-4473-be41-a4f8c0031db1)

### Overview

This library implements a scale invariant version of the original PNN proposed 
by Specht with the added functionality of allowing for smoothing along multiple
dimensions. By using a general multivariate gaussian kernel for density
estimation, the pattern units are scale invariant while accounting for
covariances within the data set. This type of neural network provides the
benefits of fast training time relative to backpropagation and statistical
generalization with only a small set of known observations. Additionally, the
Python methods utilize C++ on the backend to allow for faster processing of
large data sets.

- `Dockerfile` runs the unit tests and builds the Python library.
- `boost.cpp` contains C++ methods for testing the Boost Python library.
- `spnn.cpp` contains the backend C++ methods.
- `spnn.py` contains the Python spnn class, which can be used with scikit-learn
Pipelines.

### Setup Git Hooks
The files `.flake8`, `pyproject.toml`, and `.pre-commit-config.yaml` are used, 
in addition to the black code formatter, to autoformat code that meets code
style guidelines. e.g. line length must be <= 80. In order to use this, follow 
these steps:

1. Within the project repo, run `pre-commit install`.
2. Then run `pre-commit autoupdate`.
3. To run pre-commit git hooks for flake8 and black run use 
`pre-commit run --all-files`.