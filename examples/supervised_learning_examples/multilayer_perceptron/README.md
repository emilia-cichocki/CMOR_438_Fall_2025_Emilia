# Multi-Layer Perceptron

This folder showcases the use of the multi-layer Perceptron (MLP) on a sample dataset for classification tasks. This README contains a description of the MLP algorithm, its advantages and disadvantages, and the common evaluation metrics, as well as the dataset used and details of the code for the custom MLP class.

## MLP Algorithm
MLP is a form of supervised learning that is often used for classification problems, either binary or multi-class. It operates by updating feature weights and biases using a gradient descent algorithm and forward/backward propagation. It can be considered an extension of the [Perceptron algorithm](../perceptron/README.md) by adding multiple hidden layers for improved robustness and complexity. The MLP algorithm is an iterative process that involves initialization, a feedforward stage, error calculation, and backpropagation to update weights and biases.

### Initialization
An MLP contains $L$ hidden layers of various sizes, not including the input layer ($l = 0$) and a final output layer ($l = L + 1$). The MLP algorithm begins by randomly initializing $L + 1$ weight matrices $\mathbf{W}^l$ connecting layer $l$ to layer $l+1$ for $l = 0, ... , L$. Each matrix has a shape ($n_{l}$, $n_{l+1}$), where $n_{i}$ is the number of neurons in layer $i$. Additionally, it initializes $L + 1$ bias vectors $b^l$, each of which is a zero vector with $n_{l+1}$ elements. For a sample with an associated feature vector 

$$
\mathbf{x}_i = [x_{i1}, ..., x_{ik}]
$$

with $k = n_0$, the MLP first sets the input layer vector to be $\mathbf{a}_i^0 = \mathbf{x}_i$. 

### Feedforward Stage
For each layer $l = 0, ..., L$, the MLP performs a feedforward step by calculating the preactivation vector 

$$
\mathbf{z}_i^{l+1} = \mathbf{a}_i^{l}\mathbf{W}^{l} + \mathbf{b}^{l}
$$

The preactivation value is then fed into an activation function to produce a post-activation vector

$$
\mathbf{a}_i^{l+1} = \sigma(\mathbf{z}_i^{l+1})
$$

where $\sigma$ is the sigmoid activation function applied element-wise and given by

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The last post-activation vector $\mathbf{a}_i^{L+1}$ is the final model output for the feedforward stage. When the output layer is one neuron, this produces the probability that the sample belongs to the positive class; when it is multiple neurons, this produces a vector containing the class probabilities for each class.

### Error Signal Calculation
Once the feedforward stage is complete, the model computes the final error signal $\delta ^{L + 1}$ as

$$
\delta _i^{L + 1} = \nabla_{\mathbf{a}} C \ \odot\ \sigma'(\mathbf{z}_i^{L + 1})
$$

where $\nabla_{\mathbf{a}} C$ is the gradient of the cost function with respect to the final post-activation output and the derivative of the sigmoid activation function is

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

The cost function is a variation of mean-squared error given by

$$
C = \frac{1}{2n} \sum_{i=1}^{n} (\mathbf{a}_i^{L + 1} - \mathbf{y}_i)^2
$$

where $n$ is the number of samples, $\mathbf{a}_i^{L + 1}$ is the final post-activation output, and the encoded true targets are $\mathbf{y}_i$. 

The gradient for a single sample is given by

$$
\nabla_{\mathbf{a}} C = \mathbf{a}_i^{L + 1} - \mathbf{y}_i
$$
making the final error signal equivalent to 
$$
\delta _i^{L + 1} = (\mathbf{a}_i^{L + 1} - \mathbf{y}_i) \odot\ \sigma'(\mathbf{z}_i^{L + 1})
$$

### Backpropagation & Weight Updates
After the error signal is calculated, the weights and biases are updated through backpropagation. This involves calculating the error signal for each layer; for $l = L, ..., 1$, the error signal $\delta _i^l$ is 

$$
\delta _i^l = \delta _i^{l+1}(\mathbf{W}^l)^\top \odot \sigma'(\mathbf{z}_i^{l})
$$

For $l = L, ..., 0$, weights and biases are updated using the formulas

$$
\mathbf{W}^l := \mathbf{W}^l - \alpha (\mathbf{a}_i^{l})^\top \delta _i^{l+1}
$$

$$
\mathbf{b}^l := \mathbf{b}^l - \alpha \delta _i^{l+1}
$$

where $\alpha$ is the learning rate for the model and determines the size of each update step. A large learning rate can make training more rapid but cause overshoots, while a small learning rate may not be sufficient to reach a minimum. 

The process of feedforward, error calculation, and backpropagation continues until the maximum number of epochs have been reached, which iteratively updates the weights and biases to minimize error and produce more accurate classifications.

### Classification Process
This implementation of the MLP algorithm uses a stochastic gradient descent style update, where weights and biases are updated after a single sample. This is an efficient implementation of gradient descent, and does not require the gradient calculations across the entire training set. For each epoch, the model performs stochastic gradient descent for all samples in a random order. This process continues until the maximum number of epochs is reached.

Once the MLP has been trained on a training dataset to find optimal weights and biases, it can be used to predict the class of a previously unseen sample. For every test sample with a feature vector 

$$
\mathbf{x}_i = [x_{i1}, ..., x_{ik}]
$$

the class probabilities are calculated by performing a feedforward step using the weights, biases, and feature vector. The final post-activation value or vector is taken as the output. For binary classification problems, the sample is labeled with the positive class if this value is greater than 0.5; in multi-class problems, the class with the highest probability is selected.


## Advantages and Disadvantages
The Perceptron is a classic algorithm for binary classification, but has several fundamental limitations in its construction.

### Advantages
- Able to handle nonlinearity
- Versatile and can be extended to regression
- Strong and reliable predictions and generalizations

### Disadvantages
- Difficult to interpret
- Computationally expensive and relatively time-intensive
- Risk of overfitting, especially with few samples or features

## Evaluation
The MLP algorithm can be evaluated with standard classification metrics. The following metrics are used:

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

6. *Log Loss*: Cross-entropy loss for predicted class probabilities that accounts for the relative confidence of a prediction; for the predicted probability of a true class $y_i$ and sample $i$, $\hat{y}_{i, y_i}$

$$
L = - \frac{1}{n} \sum_{i=1}^{n} \log \hat{y}_{i, y_i}
$$

7. *Error History*: Error count calculated on the training set for every epoch, recorded and plotted to track model learning

## Code Features
The MLP is implemented using the custom `multilayer_Perceptron` class from the supervised learning package. The following describes the class.

1. **Multi-Layer Perceptron**:
- Implements an MLP algorithm using stochastic gradient descent with a sigmoid activation function
- Hyperparameters:
    - *layers*: list of input, hidden, and output layer sizes
    - *learning_rate*: learning rate used in Perceptron update
    - *epochs*: maximum number of epochs in Perceptron update
- Methods:
    - *fit*: fits the model on the training data and class labels
    - *prediction*: predicts the class label of each sample

## Data Used
The dataset used for evaluating the MLP algorithm is the Breast Cancer Wisconsin data, which is a standard dataset in machine learning. It contains 30 numeric features for 569 samples, of which the following eleven are relevant:
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

The MLP algorithm attempts to classify tumors into malignant or benign using sets of the continuous numeric features. More information on the particular features selected can be found in the associated notebook.
