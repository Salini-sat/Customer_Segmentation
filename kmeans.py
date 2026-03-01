# Week 8: K-Means Clustering using Kaggle Dataset

# 1. Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:\\Users\\KIIT0001\\Downloads\\archive (10)\\Mall_Customers.csv")

print("Dataset shape:", df.shape)
print("Dataset size:", df.size)

print("First 5 rows of the dataset:")
print(df.head())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8,6))

plt.scatter(X_scaled[:,0], X_scaled[:,1],
            c=df['Cluster'], cmap='viridis', s=60)

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1],
            c='red', s=300, marker='X', label='Centroids')

plt.xlabel('Scaled Annual Income')
plt.ylabel('Scaled Spending Score')
plt.title('Customer Segmentation using K-Means')
plt.legend()
plt.grid(True)
plt.show()


cluster_summary = df.groupby('Cluster')[[
    'Annual Income (k$)',
    'Spending Score (1-100)'
]].mean()

print("\nCluster Analysis:")
print(cluster_summary)

print("\nK-Means clustering completed successfully!")