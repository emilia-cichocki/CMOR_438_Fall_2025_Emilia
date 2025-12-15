# Logistic Regression

This folder showcases the use of the logistic regression algorithm on a sample dataset for classification tasks. This README contains a description of the logistic regression algorithm, its advantages and disadvantages, and the common evaluation metrics, as well as the dataset used and details of the code for the custom logistic regression class.

## Logistic Regression Algorithm
Logistic regression is a supervised machine learning technique used for binary classification. It operates by using gradient descent to update feature weights and biases, and incorporating the sigmoid function to predict the probability that a sample belongs to a particular class.

### Foundation of Gradient Descent 
The version of gradient descent used in logistic regression requires defining a cost function $C$, which quantifies the error between predicted and true classes. Binary cross-entropy loss is often used as the cost function, with the formula
$$
C = - \frac{1}{n} \sum_{i=1}^{n} \Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big]
$$
where $n$ is the number of samples, $\hat{y}_i$ is the predicted probability that a sample belongs in the positive class, and $y_i$ is the true class label. For a weight vector $\mathbf{w} = [w_1, ... w_k]$, a scalar bias term $b$, and a sample with an associated feature vector $\mathbf{x}_i = [x_{i1}, ..., x_{ik}]^\top$, the predicted probability for the sample is given by
$$
\hat{y}_i = \sigma(\mathbf{w}^\top \mathbf{x}_i + b)
$$
where $\sigma$ is the sigmoid activation function that restricts probability values from 0 to 1, given by
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
Consequently, $C$ can be considered a function of $\mathbf{w}$ and $b$. Gradient descent attempts to find the minimum of this function by repeatedly computing the gradient with respect to the weights and bias, which provides the direction of steepest increase, and moving incrementally in the opposite direction. The gradient of the cost function with respect to the weights is given by the partial derivative
$$
\frac{\partial C}{\partial \mathbf{w}} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) \mathbf{x}_i
$$
and the gradient of the cost function with respect to the bias is given by the partial derivative
$$
\frac{\partial C}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
$$
To minimize the cost function, the weights and bias must be adjusted in the direction of steepest decrease, which is the negative of the calculated derivatives. Thus, they are updated iteratively with the formula
$$
\mathbf{w} := \mathbf{w} - \alpha \frac{\partial C}{\partial \mathbf{w}}
$$
$$
b := b - \alpha \frac{\partial C}{\partial b}
$$
where $\alpha$ is the learning rate for the model and determines the size of each update step. A large learning rate can make training more rapid but cause overshoots, while a small learning rate may not be sufficient to reach a minimum. 

### Stochastic Gradient Descent
This implementation of logistic regression uses stochastic gradient descent. Rather than calculating the gradient for every sample and then performing the update step (batch gradient descent), stochastic gradient descent updates the weights and biases after a single sample, making it an efficient form of gradient descent.

Logistic regression begins by randomly initializing a vector of weights with length $k$ (number of features) and a bias term. The weights and bias are then updated using the gradient descent algorithm; for each epoch, the model performs stochastic gradient descent for all samples in a random order. This process continues until the maximum number of epochs is reached.

### Classification Process
Once the logistic regression model has been trained through stochastic gradient descent on a training dataset, it can be used to predict the class of a previously unseen sample. For every test sample with a feature vector $\mathbf{x}_i = [x_{i1}, ..., x_{ik}]^\top$, the probability that it belongs to the positive class is calculated with the formula
$$
\hat{y}_i = \sigma(\mathbf{w}^\top \mathbf{x}_i + b)
$$
where $\mathbf{w}$ and $b$ are the learned weights and biases, respectively. This outputs a probability ranging from 0 to 1; a sample is classified as belonging to the positive class if $\hat{y}_i$ is greater than a threshold (typically 0.5, but can be adjusted based on the needs of the problem).

## Advantages and Disadvantages
Logistic regression is a standard algorithm for binary classification, but has several limitations that restrict its applicability.

### Advantages
- Simple and fairly easy to interpret
- Able to output both the predicted class and predicted class probabilities
- Computationally efficient and fast training period

### Disadvantages
- Sensitive to outliers and noisy data
- Overly simplistic and not able to handle high-dimensional data or multicollinearity effectively
- Assumes a linear relationship between features and the target variable
- Cannot be used for predicting a continuous variable

## Evaluation
Logistic regression can be evaluated with standard classification metrics. The following metrics are used:

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

6. *Area Under (Receiver Operating Characteristic) Curve (AUC)*: Measure of how well the model can distinguish between binary classes, computed using the Wilcoxon rank-sum formula for the sum of positive ranks (where $\sum_{i \in \text{Pos}} R_i$ is the sum of the positive ranks, $n_\text{p}$ is the number of positive samples, and $n_\text{n}$ is the number of negative samples)
$$
\text{AUC} = \frac{\sum_{i \in \text{Pos}} R_i - \frac{n_\text{p}(n_\text{p}+1)}{2}}{n_\text{p} \cdot n_\text{n}}
$$
7. *Loss History*: binary cross-entropy loss calculated on the training set for every epoch, recorded and plotted to track model learning

## Code Features
Logistic regression is implemented using the custom `logistic_regression` class from the supervised learning package. The following describes the class.

1. **Logistic Regression**:
- Implements a logistic regression algorithm using stochastic gradient descent with a sigmoid activation function
- Hyperparameters:
    - *learning_rate*: learning rate used in gradient descent
    - *epochs*: maximum number of epochs in gradient descent
    - *threshold*: threshold probability required to classify a sample in the positive class
- Methods:
    - *fit*: fits the model on the training data and class labels
    - *prediction*: predicts the class label of each sample
    - *predict_proba*: predicts the class probabilities of each sample
    - *scoring*: computes accuracy

## Data Used
The dataset used for evaluating the logistic regression algorithm is the Breast Cancer Wisconsin data, which is a standard dataset in machine learning. It contains 30 numeric features for 569 samples, of which the following eleven are relevant:
- *Mean Radius*: continuous values, mean radius of cell nuclei in a tumor
- *Mean Texture*: continuous values, represents texture of a nucleus
- *Mean Perimeter*: continuous values, mean perimeter of cell nuclei in a tumor
- *Mean Area*: continuous values, mean area of cell nuclei in a tumor
- *Mean Smoothness*: continuous values, represents smoothness of a nucleus boundary
- *Mean Compactness*: continuous values, represents how circular a nucleus boundary is
- *Mean Concavity*: continuous values, represents how severe concave portions of a nucleus are
- *Mean Concave Points*: continuous values, mean number of concave points of cell nuclei in a tumor
- *Mean Symmetry*: continuous values, mean symmetry of cell nuclei in a tumor
- *Mean Fractal Dimension*: continuous values, represents the roughness of a nucleus boundary
- *Target*: categorical data, represents whether the tumor is malignant (0) or benign (1)

The logistic regression algorithm attempts to classify tumors into malignant or benign using sets of the continuous numeric features. More information on the particular features selected can be found in the associated notebook.