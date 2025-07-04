# Task 8: K-Means Clustering - Mall Customer Segmentation

## 📌 Objective
Apply K-Means Clustering to segment mall customers based on their *Annual Income* and *Spending Score*.

## 📂 Dataset
- File: Mall_Customers.csv
- Columns used: 
  - Annual Income (k$)
  - Spending Score (1-100)

## 📊 Steps Followed
1. Loaded the dataset using Pandas.
2. Standardized the data using StandardScaler.
3. Used *Elbow Method* to find the best number of clusters (K).
4. Applied *K-Means clustering*.
5. Added cluster labels to the original data.
6. Visualized clusters using *PCA* (for 2D plot).
7. Calculated *Silhouette Score* to check cluster quality.
8. Saved clustered data in a new CSV file.

## 📁 Output Files
- Mall_Customers_Clustered.csv: Dataset with cluster labels.
- elbow_plot.png: Shows optimal K using Elbow Method.
- cluster_plot.png: 2D cluster plot using PCA.

## ⚙ Tools Used
- Python
- Pandas
- Scikit-learn
- Matplotlib


## 📘 What I Learned
- Basics of *unsupervised learning*
- How *K-Means* works
- How to find the right number of clusters
- Evaluating clusters using Silhouette Score

---
