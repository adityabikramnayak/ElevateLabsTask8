import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Load data
path = "C://Users//KIIT//OneDrive//Desktop//ELAB//Task-8//Mall_Customers.csv"
df = pd.read_csv(path)

# Optional: Show first few rows
print(df.head())

# Select features (you can change)
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find best K
inertia = []
K_range = range(1, 11)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.grid()
plt.savefig("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-8//elbow_plot.png")
plt.show()

# Apply KMeans with chosen K (e.g., 5)
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df['Cluster'] = labels

# Save clustered data
df.to_csv("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-8//Mall_Customers_Clustered.csv", index=False)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='rainbow')
plt.title("K-Means Clustering (PCA view)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()
plt.savefig("C://Users//KIIT//OneDrive//Desktop//ELAB//Task-8//cluster_plot.png")
plt.show()

# Silhouette Score
score = silhouette_score(X_scaled, labels)
print("Silhouette Score:", score)
