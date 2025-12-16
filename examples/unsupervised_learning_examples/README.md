# Unsupervised Learning (Examples)

### Unsupervised Learning
Unsupervised learning refers to a form of machine learning that independently learns patterns in unlabeled data to discover underlying structures, groupings, and similarities. Unlike supervised learning, which requires data to have associated labels, unsupervised learning models are able to detect notable data structures without external input, making them useful in tasks where the data does not have a known organization. Unsupervised learning techniques fall into several categories: clustering (grouping data based on similarities), community detection (finding communities of densely connected data), and dimensionality reduction (transforming the data to reduce the number of features while maintaining structure).

### Contents
This folder contains example implementations for each of the custom-built unsupervised learning algorithms. It includes four algorithms distributed across the following subfolders:
1. *Community Detection (Label Propagation)*
2. *DBSCAN*
3. *K-Means*
4. *Principal Component Analysis (PCA)*

### Subfolder Contents 
Every folder for a machine learning algorithm contains a notebook that demonstrates the application of the relevant algorithm (i.e., classification or regression) on a sample dataset. The structure of notebooks is shared among all examples, and includes (where applicable):
1. Data cleaning steps (largely identical across notebooks, but repeated for modularity)
2. Implementation of the algorithm using a simple and easily visualized set of features, typically derived using dimensionality reduction
3. Evaluation of model performance with changed parameters
4. Comparison with an existing model (scikit-learn or NetworkX) to validate results.

In addition, each folder contains a README file detailing the notebook and algorithm itself, including:
1. Description of the model
2. Method of implementation
3. Advantages and disadvantages
4. Metrics used for evaluation
5. Code structure from the `unsupervised_learning` package
6. Dataset used for evaluation
