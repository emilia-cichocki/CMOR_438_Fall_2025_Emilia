# Supervised Learning Package

This package stores various supervised learning algorithms, including k-nearest neighbors, Perceptron and multilayer Perceptron, linear and logistic regression, decision or regression trees, and random forest classifiers. It contains the modules listed below, each of which includes docstrings with corresponding doctests to verify functionality.

### Modules
- *Distances* (`distances`)
    - Functions for calculating distances
    - Support for standard distance metrics (e.g., Euclidean, Manhattan)
- *Decision Trees* (`decisiontrees`)
    - Classes implementing decision and regression trees
    - Support for fitting, predicting, and printing trees
    - **Note: the algorithms for both decision and regression trees are stored in this module**
- *KNN* (`knn`)
    - Classes implementing k-nearest neighbor algorithms for classification and regression
    - Support for model fitting and predictions
- *Perceptron* (`perceptron`)
    - Classes implementing Perceptron and multi-layer Perceptron
    - Support for model fitting and predictions
    - **Note: the algorithms for both Perceptron and multi-layer Perceptron are stored in this module**
- *Ensemble Methods: Random Forest* (`randomforest`)
    - Classes implementing the random forest algorithm for classification and regression
    - Support for model fitting and predictions
- *Regression*
    - Classes implementing different regression algorithms
    - Support for model fitting, prediction, and evaluation
    - **Note: the algorithms for both linear regression and logistic regression are stored in this module**