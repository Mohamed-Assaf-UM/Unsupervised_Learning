# Unsupervised Machine Learning:
Unsupervised machine learning (ML) refers to the type of machine learning where the algorithm is trained on unlabelled data. This means that the system tries to learn the patterns and structure from the input data without any explicit output labels or targets. A typical example of unsupervised learning is **clustering**, where the goal is to group similar data points together.

### Key Concepts in Unsupervised Learning:
1. **Clustering**: One of the most common tasks in unsupervised learning. It involves dividing a set of objects into groups (or clusters) where objects in the same group are more similar to each other than to those in other groups.
    - **K-means Clustering**: A partition-based clustering algorithm. It divides data into *k* clusters based on centroids. The algorithm assigns each data point to the nearest centroid.
    - **Hierarchical Clustering**: It builds a hierarchy of clusters, either in a bottom-up approach (agglomerative) or top-down approach (divisive).

### Hierarchical Clustering:
   - **Agglomerative Clustering** (bottom-up):
     - Starts by treating each data point as its own cluster.
     - Merges the closest clusters iteratively until a single cluster (or the desired number of clusters) is formed.
     - Steps:
       1. Treat each data point as a separate cluster.
       2. Find the closest pair of clusters and merge them.
       3. Repeat until all clusters are merged into one (or as many as needed).
     - This method can be visualized through a **dendrogram**. The height of the branches indicates the similarity/distance between clusters.
   - **Divisive Clustering** (top-down):
     - Starts with all data points in one large cluster and iteratively splits them into smaller clusters.
     - It’s essentially the reverse process of agglomerative clustering.
![image](https://github.com/user-attachments/assets/7256bdd1-c158-4d76-8142-70900eac7572)

### Dendrogram and Cluster Selection:
   - A **dendrogram** is a diagram that shows the arrangement of clusters produced by hierarchical clustering. It helps determine the number of clusters by cutting the dendrogram at a certain distance (or threshold).
   - The threshold for selecting clusters is based on **Euclidean distance** or **Manhattan distance** (metrics for finding the similarity between points).
   - **How to Choose the Number of Clusters (k)**:
     - Look for the longest vertical line in the dendrogram that does not intersect with any horizontal line.
     - The number of vertical lines intersected by a horizontal cut through the dendrogram at this point gives the number of clusters.
     - Adjusting the threshold changes the number of clusters: a lower threshold results in more clusters.

### Key Differences Between K-means and Hierarchical Clustering:
- **Centroids**: K-means uses centroids to represent clusters, while hierarchical clustering does not.
- **Cluster Count**: In K-means, the number of clusters (k) is predefined. In hierarchical clustering, the number of clusters can be determined based on the dendrogram.
- **Flexibility**: Hierarchical clustering is more flexible because it doesn’t require a preset k, and it also gives insight into the structure of the data through the dendrogram.

### Additional Points to Consider:
1. **Dimensionality Reduction**: Sometimes, unsupervised learning is used for dimensionality reduction, such as with **Principal Component Analysis (PCA)**. This is often used to preprocess data before clustering.
2. **Applications of Unsupervised Learning**: Customer segmentation, anomaly detection, recommendation systems, and data compression.
3. **Scalability**: K-means is more scalable for large datasets as hierarchical clustering tends to be slower due to the iterative merging or splitting process.
