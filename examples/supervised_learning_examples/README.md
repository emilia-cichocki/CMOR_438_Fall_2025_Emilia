# Supervised Learning (Examples)

### Supervised Learning
Supervised learning refers to a form of machine learning that uses a set of data with predefined labels to learn a mapping between input-output pairs, thus allowing the model to make predictions on unseen data. In supervised learning, the model compares its predictions with the given true label, systematically adjusting the prediction parameters to reduce error and learn patterns that can be generalized to other data. Supervised learning techniques fall into two main categories: classification (predicting categorical values that function as class labels), and regression (predicting continuous variables that function as target values). 

### Contents
This folder contains example implementations for each of the custom-built supervised learning algorithms. It includes ten algorithms distributed across the following seven subfolders:
1. Decision & regression trees **(contains separate examples for both decision and regression trees)**
2. KNN **(contains separate examples for both classification and regression using KNN)**
3. Linear Regression
4. Logistic Regression
5. Multi-layer Perceptron
6. Perceptron
7. Random Forest **(contains separate examples for both classification and regression using random forests)**

### Subfolder Contents 
Every folder for a machine learning algorithm contains a notebook that demonstrates the application of the relevant algorithm (i.e., classification or regression) on a sample dataset. The structure of notebooks is shared among all examples, and includes:
1. Data cleaning steps (largely identical across notebooks, but repeated for modularity)
2. Implementation of the algorithm using a simple and easily visualized set of features
3. Evaluation of model performance with changed parameters
4. Model implementation with multiple features
5. Comparison with an existing model (usually scikit-learn) to validate results.

In addition, each folder contains a README file detailing the notebook and algorithm itself, including:
1. Description of the model
2. Method of implementation
3. Advantages and disadvantages
4. Metrics used for evaluation
5. Code structure from the `supervised_learning` package
6. Dataset used for evaluation

Where possible, algorithms have been grouped together for organization (e.g., the `decision_regression_trees` folder contains examples for both decision and regression trees, with a single README describing the underlying tree-based structure of both algorithms and the distinctions between the two).