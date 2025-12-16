# Linear Regression

This folder showcases the use of the linear regression algorithm on a sample dataset for regression tasks. This README contains a description of linear regression, its advantages and disadvantages, and the common evaluation metrics, as well as the dataset used and details of the code for the custom linear regression class.

## Linear Regression Algorithm
Linear regression is a supervised machine learning technique used for regression. It operates under the assumption that target values have a linear relationship with feature values, and attempting to find the best-fit line that describes this relationship. There are two main methods through which linear regression is implemented.

### Normal Equation
Linear regression can be performed by directly calculating the parameters that optimize fit using the normal equation, which provides a closed-form solution. For a feature matrix $\mathbf{X}$ with shape ($n_\text{samples}$, $n_\text{features}$) and a target vector $\mathbf{y}$ with shape ($n_\text{samples}$), the normal equation for the coefficients $\theta$ is

$$
\theta = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

To account for a bias or intercept term, a column of ones is often added to the feature matrix, resulting in the shape ($n_\text{samples}$, $n_\text{features} + 1$). 

The normal equation provides a simple method for determining the optimal coefficients for linear regression, and does not require an iterative process using gradient descent (described below). However, the normal equation can fail if the feature matrix is not invertible, and it is computationally expensive for large feature sets.

### Gradient Descent
Linear regression can also be performed using gradient descent, an optimization method that iteratively updates feature weights and biases until a minimum error is reached.

**Foundation of Gradient Descent**  
Gradient descent requires defining a cost function $C$, which quantifies the error between predicted and true target values. The following formula (a variation on mean-squared error) is often used as the cost function, where $n$ is the number of samples, $y_i$ is the true target value, and the predicted target value for a sample is $\hat{y}_i\$

$$
C = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2
$$

For a weight vector $\mathbf{w} = [w_1, ... w_k]$, a scalar bias term $b$, and a sample with an associated feature vector 

$$
\mathbf{x}_i = [x_{i1}, ..., x_{ik}]^\top
$$

the predicted value for the sample is given by

$$
\hat{y}_i = \mathbf{w}^\top \mathbf{x}_i + b
$$

Consequently, $C$ is implicitly a function of $\mathbf{w}$ and $b$, and can be written as 

$$
C(w, b) = \frac{1}{n} \sum_{i=1}^{n} (\mathbf{w}^\top \mathbf{x}_i + b - y_i)^2
$$

Gradient descent attempts to find the minimum of this function by repeatedly computing the gradient with respect to the weights and bias, which provides the direction of steepest increase, and moving incrementally in the opposite direction. The gradient of the cost function with respect to the weights is given by the partial derivative

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

**Stochastic Gradient Descent**  
This implementation of linear regression uses stochastic gradient descent. Rather than calculating the gradient for every sample and then performing the update step (batch gradient descent), stochastic gradient descent updates the weights and biases after a single sample, making it an efficient form of gradient descent.

Linear regression begins by randomly initializing a vector of weights with length $k$ (number of features) and a bias term, if applicable. The weights and bias are then updated using the gradient descent algorithm; for each epoch, the model performs stochastic gradient descent for all samples in a random order. This process continues until the maximum number of epochs is reached. The final weight vector provides the coefficients for each feature in the best-fit line, and the bias is taken as the intercept term.

## Advantages and Disadvantages
Linear regression is a common and standard algorithm for regression, but has several considerable limitations.

### Advantages
- Simple and fairly easy to interpret or visualize
- Computationally efficient and fast training period
- For an invertible feature matrix, can provide a direct closed-form solution

### Disadvantages
- Sensitive to outliers and noisy data
- Overly simplistic and not able to capture or account for complex relationships
- Assumes a linear relationship between features and the target variable
- Assumes that features are independent of one another

## Evaluation
Linear regression can be evaluated with standard regression metrics. The following metrics are used:

1. *Mean Absolute Error (MAE)*: Measure of the mean absolute difference between predicted ($\hat{y}$) and true ($y_i$) values

$$
\mathrm{MAE} = \frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert
$$

2. *Mean Squared Error (MSE)*: Measure of the mean squared difference between predicted and true values

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

6. *Training Error*: MSE calculated on the training set for every epoch, recorded and plotted to track model learning

## Code Features
Linear regression is implemented using the custom `linear_regression` class from the supervised learning package. The following describes the class.

1. **Linear Regression**:
- Implements a linear regression algorithm using either the normal equation or stochastic gradient descent
- Hyperparameters:
    - *method*: method used for regression ('normal', 'gradient_descent')
    - *fit_intercept*: allows for a bias term/intercept
    - *learning_rate*: learning rate used in gradient descent
    - *epochs*: maximum number of epochs in gradient descent
- Methods:
    - *fit*: fits the model on the training data and target values
    - *prediction*: predicts the target value of each sample
    - *scoring*: computes R2

## Data Used
The dataset used for evaluating the linear regression algorithm is the Breast Cancer Wisconsin data, which is a standard dataset in machine learning. It contains 30 numeric features for 569 samples, of which the following eleven are relevant:
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

The linear regression algorithm attempts to predict tumor concavity using sets of the additional continuous numeric features. More information on the particular features selected can be found in the associated notebook.
