# Single-Layer Perceptron

This folder showcases the use of the Perceptron on a sample dataset for binary classification tasks. This README contains a description of the Perceptron algorithm, its advantages and disadvantages, and the common evaluation metrics, as well as the dataset used and details of the code for the custom Perceptron class.

## Perceptron Algorithm
The Perceptron is a form of supervised learning that is generally used for binary classification problems. It operates by updating feature weights and biases using a Perceptron update rule. In its simplest form, it can be conceptualized as a single neuron model that takes in a set of features multiplied by weights and added to a bias term, and applies a sign activation function to the result to produce a binary output associated with class membership.

The Perceptron algorithm begins by randomly initializing a vector of weights $\mathbf{w} = [w_1, ... w_k]$ with length $k$ (number of features) and a bias term. For a sample with an associated feature vector $\mathbf{x}_i = [x_{i1}, ..., x_{ik}]^\top$, the Perceptron calculates the pre-activation value 
$$
z = \mathbf{w}^\top \mathbf{x}_i + b
$$
The pre-activation value is then fed into an activation function to produce a post-activation value
$$
a = \phi(z)
$$
where 
$$
\phi (z) =
\begin{cases}
1, & \text{if } z \ge 0 \\
-1, & \text{if } z < 0
\end{cases}
$$
The predicted class of the sample is denoted as $\hat{y}$, and is equal to the value of $a$ (either -1 or 1). The error is calculated as
$$
e_i = \hat{y}_i - y_i
$$
where $y_i$ is the true class of the sample. The weights and biases are then updated iteratively with the formula 
$$
\mathbf{w} := \mathbf{w} - \alpha e_i \mathbf{x}_i
$$
$$
b := b - \alpha e_i
$$
where $\alpha$ is the learning rate for the model and determines the size of each update step. A large learning rate can make training more rapid but cause overshoots, while a small learning rate may not be sufficient to reach a minimum. 

This implementation of the Perceptron algorithm uses a stochastic gradient descent style update, where weights and biases are updated after a single sample. This is an efficient implementation of the Perceptron update rule, and does not require the calculation of error across the entire training set. For each epoch, the model performs this stochastic update for all samples in a random order. This process continues until the maximum number of epochs is reached.

Once the Perceptron has been trained on a training dataset to find optimal weights and biases, it can be used to predict the class of a previously unseen sample. For every test sample with a feature vector $\mathbf{x}_i = [x_{i1}, ..., x_{ik}]^\top$, the class prediction is given by 
$$
a = \phi(\mathbf{w}^\top \mathbf{x}_i + b)
$$
where $\phi$ is the activation function, and $\mathbf{w}$ and $b$ are the learned weights and bias, respectively. This outputs a value of -1 or 1, which is then matched with the corresponding class label.

## Advantages and Disadvantages
The Perceptron is a classic algorithm for binary classification, but has several fundamental limitations in its construction.

### Advantages
- Simple and fairly easy to interpret
- Computationally efficient and fast training period
- Guaranteed convergence for linearly separable data
- Foundation for complex models (e.g., [multi-layer Perceptron](../multilayer_perceptron/README.md))

### Disadvantages
- Sensitive to outliers and noisy data
- Only able to handle binary classification
- Not guaranteed to converge if data is not linearly separable, and cannot solve problems that are not linear in nature

## Evaluation
The Perceptron can be evaluated with standard classification metrics. The following metrics are used:

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

6. *Error History*: Error count calculated on the training set for every epoch, recorded and plotted to track model learning

## Code Features
The Perceptron is implemented using the custom `Perceptron` class from the supervised learning package. The following describes the class.

1. **Perceptron**:
- Implements a standard Perceptron algorithm using the Perceptron update rule with a sign activation function
- Hyperparameters:
    - *learning_rate*: learning rate used in Perceptron update
    - *epochs*: maximum number of epochs in Perceptron update
- Methods:
    - *fit*: fits the model on the training data and class labels
    - *prediction*: predicts the class label of each sample
    - *scoring*: computes accuracy

## Data Used
The dataset used for evaluating the Perceptron algorithm is the Breast Cancer Wisconsin data, which is a standard dataset in machine learning. It contains 30 numeric features for 569 samples, of which the following eleven are relevant:
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

The Perceptron algorithm attempts to classify tumors into malignant or benign using sets of the continuous numeric features. More information on the particular features selected can be found in the associated notebook.