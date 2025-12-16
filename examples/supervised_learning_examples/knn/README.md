# K-Nearest Neighbors (KNN)

This folder showcases the use of the k-nearest neighbors (KNN) algorithm on a sample dataset for both classification and regression tasks. This README contains a description of the KNN algorithm, its advantages and disadvantages, and the common evaluation metrics, as well as the dataset used and details of the code for the custom KNN class.

## KNN Algorithm
K-nearest neighbors (KNN) is a supervised machine learning algorithm that can be flexibly used for either classification or regression. It operates on the guiding principle that points similar to one another will be close in space, thus allowing for determination of the properties of a previously unseen point by examining its neighbors. 

KNN takes a set of labeled training data with $n$ numerical feature values, which are used to position each sample as a point in $n$-dimensional space. For a given unlabeled test sample with values for each of the $n$ associated features, the distance from this point to each point in the training sample is calculated with a specified distance metric. Points are then ranked based on their distance to the test sample (near to far), and the $k$-nearest points (neighbors) to the sample are selected. The label or target value of these neighbors are then used to determine the predicted label or target of the sample. In KNN classification, where each training point is labeled with a specific category (typically encoded as a discrete integer value corresponding to a class), the majority label out of the neighbors is assigned to the test sample. In KNN regression, where each training point has a target that is a continuous numerical value, the mean of the neighbor targets is assigned to the test sample.

### Distance Variations
Distances in KNN can be calculated using one of several metrics. Given two points $x = (x_1, ..., x_n)$ and $y = (y_1, ..., y_n)$ in $n$-dimensional space, the following metrics are common.

*Euclidean distance*: a standard distance metric that finds the straight-line distance between two points in Euclidean space. The formula for Euclidean distance is  

$$
    d(x, y) = \sqrt{{\sum_{i=1}^{n} (x_i - y_i)^2}}
$$

*Manhattan distance*: the total distance between the absolute difference of coordinates for two points. The formula for Manhattan distance is

$$
    d(x, y) = {\sum_{i=1}^{n} |x_i - y_i|}
$$

*Minkowski distance*: a generalization of Euclidean and Manhattan distances. The formula for Minkowski distance, where $p$ is a tunable integer parameter, is

$$
    d(x, y) = ({\sum_{i=1}^{n} |x_i - y_i|^p})^{\frac{1}{p}}
$$

### Weighting
The labels or target values of neighbors in KNN can be assigned different weights in the calculation of the test label or target, depending on the distance from the neighbor to the target. Two common weighting metrics are used.

*Uniform*: each neighbor is given equal weight, regardless of the distance between it and the test sample. In classification, the majority label is taken for the test sample; in regression, the simple mean is used to find the target variable.

*Distance*: the label or target values of neighbors are weighted based on how far the point is from the test sample. For a test point $x$ and a set $\mathcal{N}_k(x)$ of $k$ -nearest neighbor points $x_i$, the weighting for each neighbor is 

$$
w_i = \frac{1}{d(x, x_i)}
$$

For classification, the probability for a class $c$ given the class labels $y_i$ for each neighbor point is 

$$
P(y = c \mid x) =
\frac{\sum_{i \in \mathcal{N}_k(x)} w_i \, \mathbf{1}(y_i = c)}
{\sum_{i \in \mathcal{N}_k(x)} w_i}, \quad
\mathbf{1}(y_i = c) =
\begin{cases} 
1, & \text{if } y_i = c \\ 
0, & \text{otherwise} 
\end{cases}
$$

The class with the maximum probability is assigned as the label to the test sample.

For regression, the weighted average of the target values $y_i$ for a set of $i$ neighborhood points is assigned as the value of the test sample through the formula

$$
\hat{y}(x) = \frac{\sum_{i \in \mathcal{N}_k(x)} w_i \, y_i}{\sum_{i \in \mathcal{N}_k(x)} w_i}
$$

## Advantages and Disadvantages
KNN is a versatile algorithm with several distinct advantages, but has limitations in its scope and robustness.

### Advantages
- Easy to implement
- No training period (training data is stored for direct calculations)
- Adaptable for both classification and regression
- Non-parametric (can handle data that is nonlinear, does not follow a normal distribution, etc.)

### Disadvantages
- Slow and computationally expensive calculations of pairwise distance
- Sensitive to noisy points that might skew distance calculations
- Poor performance in higher dimensions and as distance between points increases 

## Evaluation
KNN can be evaluated with either standard classification or regression metrics. 

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
1. *Mean Absolute Error (MAE)*: Measure of the mean absolute difference between predicted ($\hat{y}$) and true ($y_i$) values

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

5. *Adjusted R2*: Calculation of R2 that accounts for feature number to penalize unnecessary additions

$$
R^2_{\text{adj}} =
1 - \left(1 - R^2\right)
\frac{n - 1}{n - p - 1}
$$

## Code Features
KNN is implemented using the custom `knn_classification` and `knn_regressor` classes from the supervised learning package. The following describes each class.

1. **KNN Classification**:
- Implements a standard KNN classification algorithm
- Hyperparameters:
    - *k*: number of neighbors considered
    - *metric*: metric used for distance calculations ('euclidean', 'manhattan', or 'minkowski')
    - *weight*: weighting used ('uniform', 'distance')
    - *p*: distance parameter (if using Minkowski distance)
- Methods:
    - *fit*: fits the model on the training data and labels
    - *probabilities*: returns the probability for each class
    - *prediction*: predicts the class of each sample
    - *scoring*: computes accuracy

2. **KNN Regressor**:
- Implements a standard KNN regression algorithm
- Hyperparameters:
    - *k*: number of neighbors considered
    - *metric*: metric used for distance calculations ('euclidean', 'manhattan', or 'minkowski')
    - *weight*: weighting used ('uniform', 'distance')
    - *p*: distance parameter (if using Minkowski distance)
- Methods:
    - *fit*: fits the model on the training data and target values
    - *prediction*: predicts the target value of each sample
    - *scoring*: computes R2

## Data Used
The dataset used for evaluating the KNN classifier and regressor is the Palmer Penguins data, which is a standard toy dataset in machine learning. It contains the following feature data for 344 samples:
- *Species*: categorical data, one of 'Adelie', 'Chinstrap', or 'Gentoo'
- *Island*: categorical data, one of 'Biscoe', 'Dream', or 'Torgersen'
- *Bill Length (mm)*: continuous values, measuring bill length in millimeters
- *Bill Depth (mm)*: continuous values, measuring bill depth in millimeters
- *Flipper Length (mm)*: continuous values, measuring flipper length in millimeters
- *Body Mass (g)*: continuous values, measuring body mass in grams
- *Sex*: categorical data, one of 'Male' or 'Female'  

The KNN classifier attempts to classify species using sets of the continuous numeric features, while the KNN regressor attempts to predict the flipper length of a sample given additional feature data. More information on the particular features selected can be found in each of the associated notebooks.
