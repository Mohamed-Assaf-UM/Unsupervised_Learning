# Unsupervised Machine Learning:
Unsupervised machine learning is a type of machine learning where the model is not provided with labeled data or specific output to predict. The primary goal is to find hidden patterns or intrinsic structures in the input data. Unlike supervised learning, there are no target variables or dependent features in unsupervised learning. The key points in unsupervised machine learning include:

### Key Concepts
1. **No Labeled Data**: In supervised learning, you have input features (independent variables) and a corresponding output feature (dependent variable), but in unsupervised learning, there are no labels or specific outputs to predict.
2. **Clustering**: The most common type of problem solved by unsupervised learning is clustering. The idea is to group similar data points together based on their features.
3. **Examples of Features**: If you have features like age, years of experience, and salary, you won't be predicting the salary as in supervised learning. Instead, you aim to group people based on similarities in these features (e.g., age and salary).
4. **Real-World Use Case: Customer Segmentation**: 
   - This involves dividing customers into groups (or clusters) based on shared characteristics like purchasing habits, spending scores, or preferences.
   - Companies can use this technique to target specific groups, offering personalized promotions or discounts. For example, frequent buyers may get a 15% discount on new products, while occasional buyers could receive a 20% discount to encourage them to purchase.

### Algorithms Covered
1. **K-Means Clustering**: This algorithm partitions the data into `k` clusters by minimizing the distance between data points and their respective cluster centroids.
2. **Hierarchical Clustering**: It builds a hierarchy of clusters either through a bottom-up (agglomerative) or top-down (divisive) approach.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: This algorithm clusters data points based on density and can handle noise in the data effectively.
4. **Silhouette Scoring**: This score evaluates the quality of clusters by measuring how similar a data point is to its own cluster compared to other clusters. It helps validate the clustering results.

### Additional Points
- **Dimensionality Reduction**: Another common unsupervised learning task is dimensionality reduction, where algorithms like PCA (Principal Component Analysis) are used to reduce the number of input features while preserving the structure of the data. This can be useful for visualization and reducing computational complexity.
- **Association Rule Mining**: Algorithms like Apriori are used in unsupervised learning to find relationships between variables in large datasets. This is often used in market basket analysis to discover patterns in customer purchases.
- **Anomaly Detection**: Unsupervised learning is also widely applied in detecting outliers or anomalies in datasets. For instance, in cybersecurity, unsupervised algorithms can identify unusual activity that may signal a security breach.

Unsupervised learning allows us to explore data, find patterns, and segment it without the need for predefined labels, which makes it highly useful in domains where labeled data is scarce or unavailable.

---

## Principal Component Analysis (PCA)
This explanation introduces the concept of **Principal Component Analysis (PCA)** and addresses the problem it seeks to solve: the **curse of dimensionality**. Let's break it down more clearly:

### What is PCA?
PCA is a **dimensionality reduction** technique that helps reduce the number of features in a dataset while retaining the essential information. It transforms the original features into a smaller set of new features, called **principal components**, which capture most of the variance (information) in the data.

### Why Use PCA?
The key reason for using PCA is to combat the **curse of dimensionality**, which occurs when a dataset has too many features (dimensions). This often leads to:
- **Overfitting**: When the model learns noise or irrelevant features, it can perform poorly on new, unseen data.
- **Model Confusion**: With too many features, the model can become confused and make inaccurate predictions.
- **Computational Complexity**: More dimensions lead to more complex and time-consuming calculations.

### Example of Curse of Dimensionality
Imagine you're trying to predict the price of a house using several features (size, number of bedrooms, bathrooms, etc.). As you add more features, some of them may be irrelevant, and too many features can overwhelm the model, leading to worse performance:
- Initially, adding important features (like house size and bedrooms) improves the accuracy.
- But as you add irrelevant or less important features, the model's accuracy starts decreasing because it overfits and tries to learn unnecessary information.

### Solutions to the Curse of Dimensionality
1. **Feature Selection**: Choose only the most important features and discard the rest.
2. **Dimensionality Reduction (PCA)**: Instead of discarding features, PCA combines them into new features (principal components) that capture the same important information but with fewer dimensions.

In PCA, you don't directly discard features; you **extract new features** (components) that summarize the original data in fewer dimensions. This helps maintain the model's performance while reducing complexity.

### Key Takeaways:
- The **curse of dimensionality** is when increasing features leads to reduced model performance.
- **PCA** helps by reducing the number of features while keeping the essential information intact.
- PCA is particularly useful when you have many features, some of which may not contribute much to the predictive power of your model.

---
### Feature Selection vs. Feature Extraction

In machine learning, reducing the dimensionality of data is crucial for improving model performance and avoiding overfitting. Two main techniques for dimensionality reduction are **feature selection** and **feature extraction**, both aimed at simplifying the dataset while retaining the most important information.

#### Why Perform Dimensionality Reduction?
- **Prevent Curse of Dimensionality**: As the number of features (dimensions) increases, the data becomes sparse, which can lead to overfitting, higher computational costs, and poor model generalization.
- **Improve Model Performance**: Reducing features makes training models faster, as the algorithm has fewer parameters to compute.
- **Data Visualization**: Humans can only visualize in 2D or 3D. Reducing dimensions helps in understanding the data by creating 2D or 3D plots for analysis.

#### 1. Feature Selection
Feature selection is the process of identifying and selecting the most relevant features from the original dataset that contribute significantly to the prediction of the target variable.

**Key Points:**
- It eliminates irrelevant or redundant features.
- The goal is to keep the most informative features that have a strong relationship with the target variable.

**Example**:
- Consider a housing dataset with features like `house_size` and `fountain_size`. If `fountain_size` is found to have little or no correlation with the house price, it can be removed, as it doesn't contribute significantly to the prediction of house price.

**Techniques Used in Feature Selection**:
- **Covariance**: Measures how two variables vary together. Positive covariance indicates a direct relationship, while negative covariance indicates an inverse relationship.
  - A covariance of zero suggests no relationship between the variables.
- **Correlation (Pearson Correlation)**: Standardized form of covariance, giving a value between -1 and +1, where:
  - +1 indicates strong positive correlation.
  - -1 indicates strong negative correlation.
  - 0 indicates no correlation.
  
  Pearson correlation helps identify whether features have a strong linear relationship with the target.
![image](https://github.com/user-attachments/assets/32534572-f656-49e8-8082-b99d39a6c5eb)

#### 2. Feature Extraction
Feature extraction is the process of transforming the original features into a new set of features, often fewer in number, that still capture the relevant information for predicting the target variable.

**Key Points:**
- Instead of selecting from existing features, new features are **derived** by combining or transforming the original features.
- This technique is useful when the original features are important, but a smaller number of composite features can still provide sufficient predictive power.

**Example**:
- Suppose a housing dataset has two features: `room_size` and `number_of_rooms`. Both are important for predicting the house price, but instead of using them separately, they could be combined into a single feature, `house_size`, which still retains the critical information needed for prediction.

**How Feature Extraction Works**:
- The key idea is to apply **transformations** to the original data to create new features.
- For instance, **Principal Component Analysis (PCA)** is a popular technique for feature extraction, where the original features are transformed into principal components that capture the maximum variance in the data.

#### Summary
- **Feature Selection**: Involves selecting the most important features by evaluating their relationship with the target variable (e.g., through correlation or covariance).
- **Feature Extraction**: Involves creating new features by transforming or combining existing features to retain relevant information while reducing dimensionality.

---

### What is PCA?

PCA is a technique used to **reduce the number of features** (dimensions) in a dataset while keeping as much of the original data's variance (information) as possible. It does this by finding **new directions (axes)** called **principal components** that capture the maximum spread (variance) of the data.
![image](https://github.com/user-attachments/assets/8c8e0a4d-da1a-4857-954e-c8d8afa9f2d9)

### Example:
Let’s say we have a dataset with **two features**:
1. **Size of the house** (in square feet)
2. **Number of rooms**

We want to reduce this to **one feature** that captures the most information from both.

### Step-by-Step Process:

#### 1. Visualizing the Data:
Let’s plot the data with "size of the house" on the X-axis and "number of rooms" on the Y-axis.

Imagine that the points are spread in such a way that **as the size of the house increases, the number of rooms also increases**. The data points are not perfectly aligned along either the X-axis (size) or the Y-axis (rooms), but they form a diagonal pattern across both axes.

#### 2. Simple Projection (What happens if we reduce without PCA):
If we simply **project** all the points onto the X-axis (size of the house), we would lose all the information about the **number of rooms**. So, while we have reduced the data to one dimension (X-axis), we lose a lot of information, which can negatively impact any models we build.

#### 3. PCA's Solution:
Instead of using the original axes (size and rooms), **PCA finds a new direction (a new axis)** that best captures the spread of the data (variance). This new axis is called the **first principal component (PC1)**.

- **PC1**: It’s a line (axis) that captures the most variance. For example, it could be a diagonal line through the data points where the relationship between size and number of rooms is best represented.
- **PC2**: The second principal component (PC2) would be perpendicular to PC1 and would capture any remaining variance. However, it usually captures less variance than PC1.

#### 4. Dimensionality Reduction:
Now that we have these new axes (PC1 and PC2), we can **project** our data onto **PC1**, which captures the maximum variance. By doing this:
- We **reduce** the data from 2D (size and rooms) to **1D** (PC1).
- We retain most of the important information (variance) from both features, without losing much.

#### 5. Example in Practice:
Imagine after PCA, we find that **PC1** represents a combination of "size of the house" and "number of rooms" that best explains the data.

For example:
- PC1 could represent a new feature like “**house capacity**,” which is a combination of both size and number of rooms, because they are strongly related.

By projecting onto PC1, we now have a **single feature** that captures most of the important information, rather than working with two separate features (size and rooms).

### Why Use PCA?
- **Dimensionality reduction**: It helps when you have too many features and want to simplify the dataset.
- **Less information loss**: PCA finds the best directions that preserve as much data as possible.
- **Improved model performance**: Reducing the dimensions while keeping the important information can lead to simpler, faster, and more effective machine learning models.

### Summary:
- PCA helps reduce features (dimensions) while keeping the maximum variance (information).
- It creates **new axes (PC1, PC2, etc.)** to capture the spread of data.
- **PC1** captures the most important feature of the data.
- You reduce the data along PC1, losing very little information compared to a simple projection on the original axes.

By using PCA, we avoid losing important data like the number of rooms while focusing on the most meaningful features for analysis.

This is the essence of PCA: finding the best new directions (principal components) that capture the data’s spread, allowing us to reduce dimensions without losing much information.

Sure! Let’s rewrite the explanation with a simple example to make it easier to understand. We’ll go step by step and keep things straightforward.

### **Understanding PCA with an Example**

Let’s say you have a simple dataset with two features: **height (X)** and **weight (Y)**. You want to reduce this data from two dimensions to one dimension while capturing as much information as possible. Here's how PCA helps you achieve that.

### 1. **Projections**

Imagine a graph where the **x-axis** represents height and the **y-axis** represents weight. Each data point represents a person’s height and weight, plotted on this 2D plane.

- The goal of PCA is to find a new **axis (principal component)** where the data shows the most variation (spread).
- Let’s say after plotting the points, you notice that the points are spread out diagonally. PCA will rotate the axes to align with this diagonal spread because it captures the maximum variance in the data.
  
#### Example:
Consider a single data point, **P1 (Height = 150 cm, Weight = 60 kg)**, plotted on the graph.
- PCA will "project" this point onto the new axis (let’s call it PC1), which is aligned with the diagonal direction where the data is most spread.
- If you project **P1** onto this new axis, you get a new point **P1'** along this axis.

**Mathematical projection**: The formula to project **P1** onto a new unit vector (PC1) is:
![image](https://github.com/user-attachments/assets/cbc081b9-10c5-4689-abda-02c22f42f32c)

Where:
- **P1 = (150, 60)** (height and weight).
- **u** is the unit vector along the new principal component direction.
- The result, **P1'**, is the new value for **P1** on this axis (it’s now a 1D value rather than a 2D coordinate).

### 2. **Maximizing Variance (Cost Function)**

PCA aims to find the new axis (PC1) where the variance (spread) of the data is the **largest**. This means we want the data points to be as spread out as possible when projected onto this new axis.

- Why maximize variance? Because more spread means more information is retained when reducing dimensions. If all the points cluster together on the new axis, you lose information.
  
#### Example (continued):
After projecting all the data points (like **P2, P3, … Pn**) onto PC1, you get a set of new values: **P1', P2', P3', …, Pn'**.
![image](https://github.com/user-attachments/assets/2a78787d-ab3a-4a03-8ea0-8e78e15b3800)

The **cost function** is to **maximize this variance**. The more spread out the projected points are, the better the principal component (PC1) is.

### 3. **Eigenvectors and Eigenvalues**

Now, to efficiently find the best new axis (principal component), PCA uses something called **eigenvectors** and **eigenvalues**.

#### Step-by-Step Process:
1. **Compute the Covariance Matrix**:
![image](https://github.com/user-attachments/assets/a75b7152-d069-4acc-bbba-c12a79300537)


2. **Find Eigenvectors and Eigenvalues**:
   - **Eigenvectors** represent the **directions** (axes) where the data varies the most.
   - **Eigenvalues** represent the **amount of variance** captured by each eigenvector.
   
![image](https://github.com/user-attachments/assets/ecce1eaf-c1e6-454d-a93c-4ff23801384b)


3. **Select the Best Principal Component**:
   - The eigenvector with the **largest eigenvalue** is chosen as the first principal component (PC1). This is the direction where the data shows the maximum variance.
   - For example, if the eigenvalue for the diagonal direction (PC1) is larger than for the other directions, PC1 is chosen because it captures the most variance.

### Example with Eigenvalues:
- Let’s say for your height-weight dataset, you compute the covariance matrix, and you get two eigenvectors:
  1. Eigenvector 1 (PC1) captures 90% of the variance.
  2. Eigenvector 2 (PC2) captures 10% of the variance.
  
  In this case, you’ll choose **PC1** as the new axis because it captures most of the data’s variance.

### 4. **Projecting Data onto Principal Components**

After choosing PC1, you project all your data points onto this new axis. This reduces the data from 2D (height and weight) to 1D, while still keeping the most important information (since PC1 captures the maximum variance).

#### Final Example:
Imagine your dataset originally looks like this:

| Height (cm) | Weight (kg) |
|-------------|-------------|
| 150         | 60          |
| 160         | 65          |
| 170         | 70          |
| 180         | 75          |

After applying PCA, your data might look like this (projected onto PC1):

| PC1 (New Axis) |
|----------------|
| 1.5            |
| 2.0            |
| 2.5            |
| 3.0            |

Now, you have reduced the data to one dimension while keeping most of the original information.

### **Steps of PCA in Summary**:
1. **Standardize the data**: Scale features so they have equal importance.
2. **Calculate the covariance matrix**: Understand how features vary together.
3. **Find eigenvectors and eigenvalues**: Identify the directions of maximum variance.
4. **Select the principal component**: Choose the eigenvector with the largest eigenvalue (it captures the most variance).
5. **Project the data**: Transform the original data onto this new axis to reduce dimensions.

By following these steps, you reduce the dimensions of your data while preserving as much information (variance) as possible.

---

### Understanding Eigenvectors and Eigenvalues: An Example-Based Explanation
![image](https://github.com/user-attachments/assets/abefbdc9-c26b-419b-9fd8-1adf97d28baf)

Let's break down eigenvectors and eigenvalues with examples to make it easier for you to understand, particularly focusing on their role in **Principal Component Analysis (PCA)**.

---

### Step-by-Step Explanation with Example:

#### 1. **Objective of Eigenvectors and Eigenvalues in PCA**:
Eigenvectors and eigenvalues help us find the best "principal component" or the direction (line) along which the data has the most variance. The idea is that we want to reduce the number of dimensions (for example, from 3D to 2D) while still keeping as much information (variance) as possible.

#### 2. **What is Linear Transformation?**
A linear transformation is when you move or stretch data. Think of it as transforming a grid system. If you had a grid of points (your data) on a 2D plane, a linear transformation could shift, rotate, or stretch that grid. Applying a matrix to a vector (which is a point in the data) transforms that vector to a new position.

**Example**:
![image](https://github.com/user-attachments/assets/8c9cb9f7-d7e5-4314-b318-4ceb8cda8fa9)

This means the vector `[1, 1]` has now been stretched to `[2, 3]`. The new vector `[2, 3]` represents how the vector is transformed in the new space.

#### 3. **Eigenvalue and Eigenvector**:
An eigenvector is a vector that doesn’t change its direction when a transformation is applied to it, only its magnitude is scaled. The scaling factor is called the eigenvalue.

![image](https://github.com/user-attachments/assets/1700ab55-efd6-4959-af54-acc2229c382d)

This shows that the vector `[1, 0]` is scaled by 3, making `λ = 3`, and the direction of the vector hasn't changed.

#### 4. **Finding the Principal Component**:
In PCA, after we calculate eigenvalues and eigenvectors for the data’s covariance matrix, the eigenvector with the highest eigenvalue becomes the principal component because it captures the most variance in the data.

**Example**:
Imagine you have a dataset with 2 features `X` and `Y`. After calculating the covariance matrix, you find two eigenvalues, `λ1 = 5` and `λ2 = 2`. The eigenvector associated with `λ1 = 5` will be the direction that captures the most variance, making it the first principal component (PC1).

#### 5. **Steps in PCA**:

1. **Standardize the Data**:
   - We subtract the mean from each feature to ensure the data is zero-centered.

   Example: If you have data points like `X = [2, 4, 6]`, after standardization, the mean is subtracted:
   - `X_centered = [2-4, 4-4, 6-4] = [-2, 0, 2]`

2. **Covariance Matrix**:
   - Calculate the covariance matrix of the standardized data. This matrix shows how much the features vary together.

   ![image](https://github.com/user-attachments/assets/6bba9b47-542a-4c8b-ac70-4749b4d4ad24)


3. **Find Eigenvectors and Eigenvalues**:
   - Compute eigenvectors and eigenvalues of the covariance matrix. These give us the directions (eigenvectors) and the importance (eigenvalues) of those directions.

4. **Select Principal Components**:
   - Choose the eigenvector with the highest eigenvalue as PC1 (it captures the most variance).

5. **Project Data**:
   - Project the original data onto the principal component line (eigenvector), effectively reducing the dimensionality.

   Example: If `λ1 = 5` and `λ2 = 2`, we choose the eigenvector associated with `λ1` as the main component, and project the data onto that line.

---

### Simplified Recap:
- **Eigenvector**: Direction along which the data varies the most.
- **Eigenvalue**: The magnitude of variance in that direction.
- **PCA Goal**: Find the direction (principal component) that captures the most variance and project the data onto that line to reduce dimensions.

This process helps simplify high-dimensional data into fewer dimensions while retaining the most important information (variance).
