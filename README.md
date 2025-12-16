# CMOR 438 Fall 2025 - Data Science & Machine Learning
*Author*: Emilia Cichocki  
*Last Updated*: December 2025

## Description
This repository hosts a collection of custom-built machine learning algorithms for unsupervised (e.g., clustering and community detection) and supervised learning (e.g., classification and regression) machine learning tasks. Additionally, it includes a variety of functions intended to facilitate data preprocessing, postprocessing, and algorithm evaluation. 

In addition to the source code, the functionality of each algorithm is directly demonstrated by applying it to an example dataset and evaluating the results. Descriptions of the algorithm itself and the implementation can be found in the README files housed in the respective folders under [*examples*](examples/README.md).

## Contents
The contents of the repository are listed below. More information on each section, if applicable, is given in the linked README.
1. *Algorithm Examples*
    - Each machine learning algorithm is showcased in [`examples`](examples/README.md)
2. *Source Code Modules*
    - Supervised machine learning algorithms are in [`src/rice_ml/supervised_learning`](src/rice_ml/supervised_learning/README.md)
    - Unsupervised machine learning algorithms are in [`src/rice_ml/unsupervised_learning`](src/rice_ml/unsupervised_learning/README.md)
    - Preprocessing functions are housed in [`src/rice_ml/preprocess`](src/rice_ml/preprocess/README.md)
    - Postprocessing functions are housed in [`src/rice_ml/postprocess`](src/rice_ml/postprocess/README.md)
3. *Unit and Integration Tests*
    - To verify functionality, unit and integration tests are included for each algorithm in `tests`

## Dependencies & Requirements
The code in this repository has been built and tested using Python 3.9.6. The following Python packages are required (specific versions known to work with the code are listed in each of the associated notebooks)
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Sklearn (used for model comparison, but not creation)
- Torch (used for loading datasets)