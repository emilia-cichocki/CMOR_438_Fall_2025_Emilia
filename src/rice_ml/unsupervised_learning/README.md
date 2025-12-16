# Unsupervised Learning Package

This package stores various unsupervised learning algorithms, including principal component analysis (PCA), k-means clustering, DBSCAN, and community detection using label propagation. It contains the modules listed below, each of which includes docstrings with corresponding doctests to verify functionality.

### Modules
- *Clustering* (`clustering`)
    - Classes implement various clustering algorithms
    - Support for model fitting on unlabeled data
    - **Note: the algorithms for both k-means and DBSCAN are stored in this module**
- *Community Detection: Label Propagation* (`communitydetection`)
    - Class implementing community detection using unsupervised label propagation
    - Support for model propagation and displaying results
- *PCA*
    - Class implementing principal component analysis using eigendecomposition
    - Support for model fitting and data transformation