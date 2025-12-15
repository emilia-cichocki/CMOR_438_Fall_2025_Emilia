# Density-Based Spatial Clustering of Applications with Noise (DBSCAN)

This folder showcases the use of the DBSCAN algorithm on a sample dataset for clustering tasks. This README contains a description of the DBSCAN algorithm, its advantages and disadvantages, and the common evaluation metrics, as well as the dataset used and details of the code for the custom DBSCAN class.

## DBSCAN Algorithm
DBSCAN is an unsupervised machine learning technique used for clustering analysis. It operates on the guiding principle that clustering can be performed by determining dense groups of points that are separated by less dense areas. Consequently, it finds groups of points that are closely packed together, and marks points in regions with lower density as noise.

DBSCAN takes a set of unlabeled data with $n$ numerical feature values and positions each as a point in an $n$-dimensional space. For each point, the points that are within some distance $\epsilon$ are defined as neighbors and counted. Distance is calculated using Euclidean distance, a standard distance metric that finds the straight-line distance between two points in Euclidean space. The formula for Euclidean distance is  
$$
    d(x, y) = \sqrt{{\sum_{i=1}^{n} (x_i - y_i)^2}}
$$

Once the set of neighbors for each point has been identified, core points are determined by finding which points have at least *k* points in their neighborhood. The algorithm then iterates across the group of points. When the first core point is found, it and all core points in its neighborhood are added to a cluster. This process continues recursively, with all core points within the neighborhood of newly clustered core points added to the same cluster. Next, the cluster is extended to all points that are not core points, but that reside within the neighborhood of a core point within the cluster (density-reachable points).

The DBSCAN algorithm iterates through each point in the set to determine whether it has or has not been previously visited and added to a cluster. When a core point is found that has not been assigned to any pre-existing cluster, the process begins again with this point. When all core points have been assigned to a cluster, any remaining points are labeled as noise.

## Advantages and Disadvantages
DBSCAN is a common and versatile clustering algorithm, but has several notable drawbacks.

### Advantages
- Does not require the number of clusters to be specified
- Able to find clusters with an arbitrary shape
- Handles outliers and noisy points automatically

### Disadvantages
- Resuls can vary significantly based on the choice of $\epsilon$ and the minimum number of neighbors required to identify a core point
- Poor performance in higher dimensions and as distance between points increases 

## Evaluation
DBSCAN can be evaluated with several clustering metrics. The following metrics are used:

1. *Number of Clusters*: Number of clusters identified
2. *Number of Noise Points*: Number of points classified as noise and not included in a cluster
3. *Cluster Counts*: Number of points per cluster
4. *Silhouette Score*: Measure of how similar a point is to its assigned cluster in comparison to other clusters
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
Silhouette scores can be implemented disregarding noise entirely, or occasionally by treating noisy points as a separate class.

## Code Features
DBSCAN is implemented using the custom `dbscan` class from the unsupervised learning package. The following describes the class.

1. **DBSCAN**:
- Implements a standard DBSCAN clustering algorithm
- Hyperparameters:
    - *epsilon*: distance used to find neighboring points
    - *min_points*: minimum number of neighbors required to be considered a core point
- Methods:
    - *fit*: fits the model on the data, identifies core points, and assigns cluster labels

## Data Used
The dataset used for evaluating the DBSCAN algorithm is the Wine data, a standard toy dataset in machine learning. Although it is typically used for classification analysis, it includes a number of numeric features suitable for unsupervised clustering. It contains the following feature data for 178 samples:
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

The DBSCAN classifier attempts to cluster points along three principal components, derived from the 13 continuous features using [Principal Component Analysis](../pca/README.md). More information on the particular features selected can be found in the associated notebook.
