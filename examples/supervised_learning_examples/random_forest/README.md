# Ensemble Methods (Random Forest)

This folder showcases the use of ensemble methods in machine learning, specifically random forest, for classification and regression tasks on a simple dataset. This README contains a description of the base random forest algorithm, its advantages and disadvantages, as well as the dataset used and details of the code for the custom class.

## Ensemble Methods
Ensemble methods are techniques used in machine learning that combine multiple models for a particular task to produce more robust results. The main methodology covered in this notebook and implemented using random forests is bagging, which trains various models on different bootstrapped samples of training data, then combines the results to produce a single output (labeling for classification, target value for regression).

Bagging as an ensemble method relies on the generation of bootstrapped data. Bootstrapping is the process of sampling a dataset to produce $N$ subsets that are the same size as the original set, and is implemented by randomly selecting samples with replacement. This allows for the repetition of items in each new set, creating an arbitrary number of data arrays that are equal in size and contain items from the training data. This bootstrapped data can be used to train $N$ smaller models, and the ability to combine these models to produce robust results forms the basis of ensemble learning.

## Random Forest
Random forest algorithms are a type of ensemble method used in supervised learning for either classification or regression. Random forests are built on combinations of decision or regression trees, each of which is trained on a bootstrapped sample of data and a randomized feature subset.

A random forest model with $N$ trees creates $N$ sets of bootstrapped data from the original training data, and trains each tree on a set (for details on decision/regression trees, see [this README](../decision_regression_trees/README.md)). For random forest classifiers, decision trees are used; for random forest regressors, regression trees are used. In addition to bootstrapping data, random forest models introduce an additional element of randomness through selecting only a subset of features to be used for training in each tree. These are chosen randomly out of the total set of features, where the size of the subset can be user-specified (typically the square root or log2 of the total number of features).

For a given test sample, the random forest algorithm works by obtaining a label (classification) or target value (regression) from each tree in the model. For a random forest classifier, the final label is chosen by taking the majority predicted label across trees. For a random forest regressor, the final target value is the average of the predicted value from each tree.

## Advantages and Disadvantages
Random forests are an accurate and common ensemble method in machine learning, with several significant limitations.

### Advantages
- Less prone to overfitting and generally results in high accuracy
- Allows for determination of feature importance
- Non-parametric (can handle data that is nonlinear, does not follow a normal distribution, etc.)
- Can handle both numerical (decision, regression) and categorical (decision) data
- Able to handle missing data and less sensitive to outliers or extreme values

### Disadvantages
- Computationally complex and can be time or memory-intensive
- Difficult to interpret, especially compared to individual trees

## Evaluation
Random forests can be evaluated with either standard classification or regression metrics. 

For classification, the following metrics are used:
1. *Accuracy*: Proportion of predictions that match true labels
2. *Precision*: Measure of how many positive predictions for a class are true positives (TP) compared to false positives (FP)

$$
P = \frac{TP}{TP + FP}
$$

3. *Recall*: Measure of how many positive instances for a class are predicted (TP) compared to false negatives (FN)

$$
R = \frac{TP}{TP + FN}
$$

4. *F1 Score*: Combination of precision and recall

$$
F_1 = 2 \cdot \frac{\text{P} \cdot \text{R}}{\text{P} + \text{R}}
$$

5. *Confusion Matrix*: Visualization of true versus predicted labels for each class

Precision, recall, and F1 can be calculated as a micro or macro average. The micro-average of precision or recall is calculated by summing the total number of true positives, false positives, and false negatives for a single calculation. The macro-average is calculated by averaging the value of the metric for each class present in the data. The micro F1 score is calculated using the micro precision and recall scores, while the macro F1 score is the average of the F1 score for each class. Thus, micro scores are heavily biased from the majority class, while macro scores provide information on the overall model ability with equal class weightings.

For regression, the following metrics are used:
1. Mean Absolute Error (MAE): Measure of the mean absolute difference between predicted ($\hat{y}$) and true ($y_i$) values

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert
$$

2. Mean Squared Error (MSE): Measure of the mean squared difference between predicted and true

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

3. Root Mean Squared Error (RMSE): Square root of MSE

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

4. Coefficient of Determination (R2): Goodness-of-fit metric that measures explained variance

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

5. Adjusted R2: Calculation of R2 that accounts for feature number to penalize unnecessary additions

$$
R^2_{\text{adj}} =
1 - \left(1 - R^2\right)
\frac{n - 1}{n - p - 1}
$$

## Code Features
Random forest models are implemented using the custom `random_forest` class from the supervised learning package. The following describes the class.

1. **Random Forest**:
- Implements a standard random forest algorithm using bootstrapping and random feature selection
- Hyperparameters:
    - *n_trees*: the number of trees in the model
    - *task*: type of model ('classification', 'regression')
    - *max_depth*: the maximum depth of each tree
    - *min_samples_split*: the minimum samples required to split a node in a tree
    - *max_features*: specifies the maximum number of features each tree is trained on (positive integer, 'sqrt', 'log2')
    - *random_state*: random state used in generation of bootstrapped data and feature selection
- Methods:
    - *fit*: fits the model on the training data and labels or target values
    - *predict*: predicts the class or target value of each sample

## Data Used
The dataset used for evaluating random forests is the Palmer Penguins data, which is a standard toy dataset in machine learning. It contains the following feature data for 344 samples:
- *Species*: categorical data, one of 'Adelie', 'Chinstrap', or 'Gentoo'
- *Island*: categorical data, one of 'Biscoe', 'Dream', or 'Torgersen'
- *Bill Length (mm)*: continuous values, measuring bill length in millimeters
- *Bill Depth (mm)*: continuous values, measuring bill depth in millimeters
- *Flipper Length (mm)*: continuous values, measuring flipper length in millimeters
- *Body Mass (g)*: continuous values, measuring body mass in grams
- *Sex*: categorical data, one of 'Male' or 'Female'  

The random forest classifier attempts to classify sex using sets of the continuous numeric features, while the regression tree attempts to predict the bill length of a sample given additional feature data. More information on the particular features selected can be found in each of the associated notebooks.
