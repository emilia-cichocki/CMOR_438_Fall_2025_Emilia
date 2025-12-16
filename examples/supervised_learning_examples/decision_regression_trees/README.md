# Decision & Regression Trees

This folder showcases the use of decision (classification) and regression tree algorithms on a sample dataset. This README contains a description of the base decision tree algorithm, its advantages and disadvantages, and the common evaluation metrics, as well as the dataset used and details of the code for the custom classes.

## Decision/Regression Tree Algorithm
Decision and regression trees are a form of supervised learning that can be used, respectively, for classification and regression tasks. They operate recursively to discover optimal feature threshold values that produce splits in the data, resulting in a flowchart or tree-like structure that is able to predict label or target variables from numerical features.

These trees are constructed with a series of root, decision, and leaf nodes, connected with decision rules represented as branches. Root nodes perform the initial division on the data by calculating the optimal split across features based on specific criteria (see below), resulting in child nodes that are either decision or leaf nodes. Each decision node contains tests for a specific feature threshold, where samples are split based on the values corresponding to that feature to create additional child nodes. This process continues until a node is produced that is not further split, denoted as a leaf node. Leaf nodes can be created either when all samples in the node are of the same class (specific to decision trees), when no beneficial feature and threshold split can be found, or when a stopping criterion (e.g., maximum tree depth or minimum samples required to split a node) has been reached.

The criteria used to split a node is found by determining the threshold value across features that produces an optimal split, which is differently defined for decision and regression trees.

### Decision Trees
Decision trees produce splits by attempting to maximize the information gain through minimizing the weighted mean entropy of the child nodes. Entropy is defined as 

$$
E = - \sum_{i=1}^{C} p_i \log_2(p_i)
$$

where $p_i$ is the proportion of samples belonging to a given class out of $C$ classes, and is a measure of the impurity of a node; nodes that contain only samples of a single class have an entropy of zero, while high entropy is indicative of samples from many different classes. Information gain is then calculated as 

$$
I = E_{\text{parent}} - \sum_{j} \frac{S_j}{S} E_j
$$

where $E_\text{parent}$ is the entropy of the parent node, $S_j$ is the number of samples in a given child node, $S$ is the number of samples in the parent node, and $E_j$ is the entropy for a child node.

For each node in a decision tree, possible thresholds across all features are calculated by finding the midpoints between consecutive unique values of a feature. For each possibility, samples are split based on whether they are less than or equal to (left node) or greater than (right node) the threshold. The information gain for the resulting split is then calculated, and the feature-threshold combination that maximizes information gain is selected. A node is considered a leaf node if it contains only samples of the same class, if no split is able to improve information gain over the parent node, or if it contains fewer samples than are required for a further split.

The process of deciding splits continues recursively for each node in the decision tree until it reaches a maximum depth, or when no further splits can be made (all child nodes from the last layer are leaf nodes). For each leaf node, the class label is decided by taking the majority label across all samples in the node. If multiple labels are tied for the majority, one is chosen arbitrarily based on class ordering.

Decision trees are able to classify test points by following the flowchart-like structure created by fitting the training data. Each sample begins at the root node, and is then placed in the left or right child node depending on whether the sample value for the node feature is less than or equal to (left) or greater than (right) threshold. This process continues until the sample reaches a leaf node, where it is assigned the label associated with the leaf.

### Regression Trees
Regression trees produce splits by attempting to maximize variance reduction through minimizing the variance of target values in the child nodes. Variance is defined as

$$
V = \frac{1}{n} \sum_{i=1}^{n} \big(y_i - \bar{y}\big)^2
$$

where $y_i$ is the target value for a sample and $\bar{y}$ is the mean target values across all samples. Variance reduction is then calculated as 

$$
\Delta \text{Var} = V_\text{parent} -
\sum_{j \in \{\text{left}, \text{right}\}} 
\frac{n_j}{n_\text{parent}} \sum_{i \in j} (y_i - \bar{y}_j)^2
$$

where $n_\text{parent}$ is the number of samples in the parent node and $n_j$ the number of samples in either the left or right node.

The process for developing a regression tree is analogous to that of a decision tree, but the optimal feature-threshold combination is chosen by maximizing variance reduction (minimizing variance) rather than maximizing information gain. For each leaf node, the target value is calculated by averaging the values for each sample in the leaf.

Regression trees are similarly able to predict target values for test points through following the flowchart structure. Each sample begins at the root node, and is then placed in the left or right child node depending on whether the sample value for the node feature is less than or equal to (left) or greater than (right) the threshold. This process continues until the sample reaches a leaf node, where it is assigned to the mean target value of samples in the leaf.

## Advantages and Disadvantages
Decision and regression trees are useful in supervised machine learning for their ease of visualization and robustness, but have several prominent drawbacks.

### Advantages
- Easy to visualize and interpret the decision process
- Allows for automatic determination of feature importance
- Non-parametric (can handle data that is nonlinear, does not follow a normal distribution, etc.)
- Can handle both numerical (decision, regression) and categorical (decision) data

### Disadvantages
- Prone to overfitting unless maximum depth and minimum samples required to split are appropriately tuned
- Can lead to sub-optimal splits due to the inherent greedy nature of the algorithm
- Somewhat unstable and sensitive to small changes in the data

## Evaluation
Decision and regression trees can be evaluated with either standard classification or regression metrics. 

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
1. *Mean Absolute Error (MAE)*: Measure of the mean absolute difference between predicted ($\hat{y}$) and true ($y$) values

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert
$$

2. *Mean Squared Error (MSE)*: Measure of the mean squared difference between predicted and true

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

3. *Root Mean Squared Error (RMSE)*: Square root of MSE

$$
\mathrm{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

4. *Coefficient of Determination (R2)*: Goodness-of-fit metric that measures explained variance

$$
R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
{\sum_{i=1}^{n} (y_i - \bar{y})^2}
$$

5. *Adjusted R2*: Calculation of R2 that accounts for number of features to penalize unnecessary additions

$$
R^2_{\text{adj}} =
1 - \left(1 - R^2\right)
\frac{n - 1}{n - p - 1}
$$

## Code Features
Decision and regression trees are implemented using the custom `decision_tree` and `regression_tree` classes from the supervised learning package. The following describes each class.

1. **Decision Tree**:
- Implements a standard decision tree algorithm using information gain and entropy
- Hyperparameters:
    - *max_depth*: the maximum depth of the tree
    - *min_samples_split*: the minimum samples required to split a node
- Methods:
    - *fit*: fits the model on the training data and labels
    - *predict*: predicts the class of each sample
    - *print_tree*: prints a visualization of the decision tree

2. **Regression Tree**:
- Implements a standard regression tree algorithm using variance reduction
- Hyperparameters:
    - *max_depth*: the maximum depth of the tree
    - *min_samples_split*: the minimum samples required to split a node
- Methods:
    - *fit*: fits the model on the training data and target values
    - *predict*: predicts the class of each sample
    - *print_tree*: prints a visualization of the regression tree

## Data Used
The dataset used for evaluating the decision trees and regression trees is the Palmer Penguins data, which is a standard toy dataset in machine learning. It contains the following feature data for 344 samples:
- *Species*: categorical data, one of 'Adelie', 'Chinstrap', or 'Gentoo'
- *Island*: categorical data, one of 'Biscoe', 'Dream', or 'Torgersen'
- *Bill Length (mm)*: continuous values, measuring bill length in millimeters
- *Bill Depth (mm)*: continuous values, measuring bill depth in millimeters
- *Flipper Length (mm)*: continuous values, measuring flipper length in millimeters
- *Body Mass (g)*: continuous values, measuring body mass in grams
- *Sex*: categorical data, one of 'Male' or 'Female'  

The decision tree attempts to classify species using sets of the continuous numeric features, while the regression tree attempts to predict the body mass of a sample given additional feature data. More information on the particular features selected can be found in each of the associated notebooks.
