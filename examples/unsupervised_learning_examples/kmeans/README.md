# K-Means Clustering

This folder showcases the use of the k-means clustering algorithm on a sample dataset for clustering tasks. This README contains a description of the k-means algorithm, its advantages and disadvantages, and the common evaluation metrics, as well as the dataset used and details of the code for the custom k-means class.

## K-Means Algorithm
K-means clustering is an unsupervised machine learning technique used for clustering analysis. It operates on the principle that a set of points in space can be partitioned in a manner that minimizes within-cluster variance, producing well-separated clusters.

K-means takes a set of unlabeled data with $n$ numerical feature values and positions each as a point in an $n$-dimensional space. For a specified number of clusters $k$, the k-means algorithm begins by randomly selecting $k$ points from the set to act as centroids. The distance between every point in the set and each centroid is calculated using Euclidean distance, a standard distance metric that finds the straight-line distance between two points in Euclidean space. The formula for Euclidean distance is  
$$
    d(x, y) = \sqrt{{\sum_{i=1}^{n} (x_i - y_i)^2}}
$$

Each point is assigned to the cluster containing the centroid that it is closest to. The centroids are then re-assigned to the mean of all points in the cluster, and the process repeats. When the summed distance from the new centroids to the previous centroids for each cluster falls below a certain tolerance, or when the maximum number of iterations has been reached, the algorithm terminates.

## Advantages and Disadvantages
K-means is a simple and common clustering algorithm, but is limited in its implementation and robustness.

### Advantages
- Easy to implement and interpret
- Computationally efficient and can be used for large datasets
- Always converges given sufficient iterations (albeit to a local optimum)

### Disadvantages
- Requires initial specification of the number of clusters
- Influenced by the random choice of initial centroids
- Does not work well for clusters that are not spherical

## Evaluation
K-means can be evaluated with several clustering metrics. The following metrics are used:

1. *Number of Clusters*: Number of clusters identified
2. *Cluster Counts*: Number of points per cluster
3. *Silhouette Score*: Measure of how similar a point is to its assigned cluster in comparison to other clusters
$$
S = \frac{1}{n} \sum_{i=1}^{n} s(i)
$$
where the silhouette score for each point $s(i)$ is given by
$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$
$b(i)$ is the minimum distance from a point in cluster $C_i$ to any other cluster $C$, given by
$$
b(i) = \min_{C \neq C_i} \frac{1}{|C|} \sum_{j \in C} \| \mathbf{x}_i - \mathbf{x}_j \|
$$
$a(i)$ is the mean distance from a point to all other points in the same cluster, given by
$$
a(i) = \frac{1}{|C_i| - 1} \sum_{\substack{j \in C_i \\ j \neq i}} \| \mathbf{x}_i - \mathbf{x}_j \|
$$
4. *Inertia (Within-Cluster Sum of Squared Distances)*: Measurement of how how well the data is clustered; the sum of the distances between all points in a cluster to the cluster centroid summed again across all clusters, given by the formula (where $C_i$ denotes clusters in the data and $c_i$ the associated centroid)
$$
I = \sum_{i=1}^{k} \sum_{x \in C_i} \| x - c_i \|^2
$$
5. *Elbow Plot*: Visualization of the inertia for different numbers of clusters, used to identify the optimal $k$

## Code Features
K-means is implemented using the custom `k_means` class from the unsupervised learning package. The following describes the class.

1. **K-Means**:
- Implements a standard k-means clustering algorithm
- Hyperparameters:
    - *n_clusters*: number of clusters
    - *max_iter*: maximum number of iterations
    - *tol*: tolerance to compare centroid shifts against
    - *random_state*: random state for initial centroid selection
- Methods:
    - *fit*: fits the model on the data, identifies centroids, and assigns cluster labels
    - *prediction*: predicts the cluster of a previously unseen data point by assigning it to the cluster of the nearest centroid (optional and not often used in practice)

## Data Used
The dataset used for evaluating the k-means algorithm is the Wine data, a standard toy dataset in machine learning. Although it is typically used for classification analysis, it includes a number of numeric features suitable for unsupervised clustering. It contains the following feature data for 178 samples:
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

The k-means algorithm attempts to cluster points along three principal components, derived from the 13 continuous features using [Principal Component Analysis](../pca/README.md). More information on the particular features selected can be found in the associated notebook.
