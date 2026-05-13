"""
Practice activity: K-means clustering on customer income/spending data.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# 3. Load the dataset
# 41 normal customers (income 15.0 -> 35.0 in 0.5 steps) plus 3 high-income
# outliers. SpendingScore is constructed to give two natural sub-groups
# within the normal customers — low spenders and high spenders.
annual_income = [
    15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
    20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
    25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
    30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
    35,
    80, 85, 90,  # outliers
]

spending_score = [
    # Low spenders (paired with the first 20 normal incomes)
    20, 25, 22, 18, 30, 28, 35, 33, 26, 31,
    24, 27, 29, 23, 21, 19, 32, 34, 36, 38,
    # High spenders (paired with the next 21 normal incomes)
    65, 70, 68, 72, 75, 80, 78, 82, 77, 83,
    85, 79, 81, 86, 88, 73, 76, 84, 87, 89, 90,
    # Outliers' spending scores
    95, 5, 50,
]

assert len(annual_income) == len(spending_score) == 44

df = pd.DataFrame({
    'AnnualIncome':  annual_income,
    'SpendingScore': spending_score,
})
print(df.head())
print(f"\nDataset shape: {df.shape}")


# 4. Preprocess: scale features so income and spending contribute equally
scaler = StandardScaler()
df_scaled_array = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled_array, columns=['AnnualIncome', 'SpendingScore'])
print("\nScaled data (head):")
print(df_scaled.head())


# 5. Implement K-means with k=3
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(df_scaled)
df['Cluster'] = kmeans.labels_

print(f"\nCluster assignments (head):")
print(df.head())
print(f"\nCluster sizes: {df['Cluster'].value_counts().sort_index().to_dict()}")
print(f"WCSS (inertia) at k={k}: {kmeans.inertia_:.4f}")


# 6. Visualize the clusters
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    df['AnnualIncome'], df['SpendingScore'],
    c=df['Cluster'], cmap='viridis', s=60, edgecolor='k', alpha=0.85,
)

# Plot the cluster centers back in original units
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(
    centers_original[:, 0], centers_original[:, 1],
    marker='X', s=220, c='red', edgecolor='black', label='Centroids',
)
ax.set_title('K-Means Clustering of Customers (k=3)')
ax.set_xlabel('Annual Income (in thousands)')
ax.set_ylabel('Spending Score (1-100)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('clusters_k3.png', dpi=120, bbox_inches='tight')
plt.show()


# 7. Elbow method — find a sensible k
wcss = []
ks = list(range(1, 11))
for i in ks:
    km = KMeans(n_clusters=i, random_state=42, n_init=10)
    km.fit(df_scaled)
    wcss.append(km.inertia_)

print("\n--- Elbow method ---")
for ki, w in zip(ks, wcss):
    print(f"  k={ki:2d}  WCSS={w:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(ks, wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('WCSS (within-cluster sum of squares)')
plt.xticks(ks)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow.png', dpi=120, bbox_inches='tight')
plt.show()


# 8. Interpretation
print("\n--- Cluster profiles (means per cluster, original units) ---")
print(df.groupby('Cluster')[['AnnualIncome', 'SpendingScore']].mean().round(2))

# Key takeaways
#
# - Always StandardScaler before K-means: it minimizes squared *Euclidean*
#   distance, so a feature with larger numeric range otherwise dominates.
# - n_init=10 (the modern default) restarts K-means with different seeds
#   and keeps the lowest-WCSS run — guards against bad initialization.
# - The elbow method is a heuristic, not a law: look for the k where the
#   marginal drop in WCSS gets noticeably smaller. On this dataset the
#   elbow sits around k=3, which matches the structure we built into the
#   data (low spenders / high spenders / outliers).
# - K-means assumes roughly spherical, equally-sized clusters. For
#   arbitrarily-shaped or density-based clusters, DBSCAN is a better fit.
