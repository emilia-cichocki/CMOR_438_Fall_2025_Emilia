# Principal Component Analysis (PCA)

This folder showcases the use of principal component analysis (PCA) on a sample dataset for dimensionality reduction. This README contains a description of the PCA algorithm, its advantages and disadvantages, and the common evaluation metrics, as well as the dataset used and details of the code for the custom PCA class.

## PCA Algorithm
PCA is an unsupervised machine learning technique that attempts to find combinations of features, denoted as principal components, that explain maximum variance in the data. It operates on the principle that dimensionality can be reduced with minimal information loss by rotating the data to find an orthogonal basis that capture variance.

### Principal Component Construction
PCA is performed by calculating the covariance matrix for input data features. For two variables $x$ and $y$ with $n$ samples each, sample covariance is defined as

$$
\text{Cov}(x, y) = \frac{1}{n - 1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})
$$

where $\bar{x}$ and $\bar{y}$ are the means of $x$ and $y$, respectively. For $p$ variables, the covariance matrix is defined as the matrix $C_{p \text{x} p}$, where each entry $C_{ij}$ denotes the sample covariance between the $i$-th and $j$-th feature.

The covariance matrix can be treated as a linear transformation that maps data onto eigenvectors. Thus, the eigenvectors of the matrix correspond to the directions of maximal variance in the data (principal components) and the eigenvalues represent the extent of variance captured along each direction.

### Data Transformation
Once the principal components are found by calculating the eigenvalues, the input data can be transformed to the newly described space, with a dimension equal to the number of principal components. For a data array $D$ with dimensions $n_\text{samples}$ by $n_\text{features}$ and a component array $P$ with dimensions $n_\text{features}$ by $n_\text{components}$, the transformed data matrix $T$ is given by

$$
T = DP
$$

where every row of $T$ represents the values for a sample on each principal component in the transformed space.

## Advantages and Disadvantages
PCA is a standard dimensionality reduction technique, but has several associated limitations.

### Advantages
- Able to reduce the dimensionality of a dataset for use in other analyses
- Results in uncorrelated principal components where multicollinearity may otherwise be an issue

### Disadvantages
- Can be difficult to interpret
- Some loss of information, especially if data is not sufficiently correlated
- Assumes linearity and cannot capture nonlinear relationships

## Evaluation
PCA can be evaluated with several quantitative and qualitative metrics. The following metrics are used:

1. *Explained Variance per Number of Components*: Visualizes the amount of variance explained by increasing the number of retained components
2. *Scree Plot*: Visualizes the eigenvalue of each principal component, often used to retain only components with eigenvalues > 1 (explains as much variance as an original feature)
3. *Component Variance*: Variance explained by each principal component

## Code Features
PCA is implemented using the custom `PCA` class from the unsupervised learning package. The following describes the class.

1. **PCA**:
- Implements a standard PCA algorithm through finding the eigenvectors of the covariance matrix
- Hyperparameters:
    - *n_components*: number of principal components
- Methods:
    - *fit*: fits the model on the data and returns eigenvalues, explained variance, and principal components
    - *transform*: transforms the data based on the calculated principal components

## Data Used
The dataset used for evaluating the PCA algorithm is the Wine data, a standard toy dataset in machine learning. Although this dataset is typically used for classification analysis, it includes a number of numeric features suitable for dimensionality reduction. It contains the following feature data for 178 samples:
- *Alcohol*: continuous values, measures alcohol content
- *Malic acid*: continuous values, measures malic acid content
- *Ash*: continuous values, measures ash content
- *Alkalinity of Ash*: continuous values, measures alkalinity of ash in the wine
- *Magnesium*: continuous values, measures magnesium content in mg/L
- *Total Phenols*: continuous values, measures total phenolic compounds
- *Flavanoids*: continuous values, measures flavonoid content
- *Nonflavanoid Phenols*: continuous values, measures nonflavonoid phenolic compounds
- *Proanthocyanins*: continuous values, measures proanthocyanidin content
- *Color Intensity*: continuous values, measures intensity of color
- *Hue*: continuous values, measures hue of color
- *od280/od315 of Diluted Wines*: continuous values, measures properties of phenolic compounds
- *Proline*: continuous values, measures proline content
- *Target*: categorical data, indicates the cultivar (one of 0, 1, 2)

PCA attempts to reduce the dimensionality of the non-categorical data to a series of principal components derived from the 13 continuous features. More information on the particular features selected can be found in the associated notebook.
