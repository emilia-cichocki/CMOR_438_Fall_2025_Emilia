# Preprocessing Package

This package can be used to preprocess raw data to a form that is usable in an implemented machine learning algorithm. It contains the modules listed below, each of which includes docstrings with corresponding doctests to verify functionality.

### Modules
- *Cleaning* (`cleaning`)
    - Functions for cleaning the data
    - Support for standard data cleaning processes (e.g., outlier detection, missing data)
- *Type* (`datatype`)
    - Functions for converting data into a form usable for machine learning algorithms
    - Support for multiple type and shape-check functions (e.g., numeric data, array sizes)
- *Split* (`split`)
    - Functions for splitting data
    - Support for finding training, testing, and validation datasets
- *Standardize* (`standardize`)
    - Functions for standardizing data
    - Support for a range of standardization metrics (e.g, z-score, min-max)