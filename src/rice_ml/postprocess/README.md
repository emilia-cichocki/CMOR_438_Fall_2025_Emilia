# Postprocessing Package

This package can be used for postprocessing data once a supervised or unsupervised algorithm has been applied. It contains the modules listed below, each of which includes docstrings with corresponding doctests to verify functionality.

### Modules
- *Classification* (`classificationpost`)
    - Functions for analyzing classification results
    - Support for calculating a variety of numeric scores (e.g., accuracy, precision, recall) and facilitating visualization (e.g., confusion matrix plotting)
- *Regression* (`regressionpost`)
    - Functions for analyzing regression results
    - Support for calculating a variety of numeric scores (e.g., MAE, MSE, R2) and displaying results (e.g., printing model metrics)
- *Unsupervised Algorithms* (`unsupervised`)
    - Functions for analyzing cluster results from unsupervised learning
    - Support for calculating a variety of numeric scores (e.g., silhouette score, label counts) and displaying results (e.g., printing model metrics)