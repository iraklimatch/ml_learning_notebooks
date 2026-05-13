"""
Practice activity: K-means vs DBSCAN — side-by-side comparison.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques

Same dataset, same preprocessing, two algorithms. The point is to see
how each one *thinks* about the same data: K-means partitions into k
fixed groups (every point gets assigned); DBSCAN groups by density and
calls outliers noise.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler


# 3. Load the dataset (same as the standalone DBSCAN activity)
annual_income = [
    15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
    20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
    25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
    30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
    35,
    80, 85, 90,  # outliers
]

spending_score = [
    39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
    68, 70, 72, 73, 75, 76, 78, 79, 80, 82,
    83, 84, 85, 86, 87, 87, 88, 88, 89, 89,
    90, 91, 92, 93, 94, 95, 95, 96, 97, 98,
    99,
    40, 60, 80,  # outliers' spending scores
]

df = pd.DataFrame({
    'AnnualIncome':  annual_income,
    'SpendingScore': spending_score,
})
print(df.head())
print(f"\nDataset shape: {df.shape}")


# 4. Preprocess: scale features so neither dominates the distance metric
scaler = StandardScaler()
df_scaled_array = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled_array, columns=['AnnualIncome', 'SpendingScore'])
print("\nScaled data (head):")
print(df_scaled.head())


# 5. K-means with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(df_scaled)
df['KMeans_Cluster'] = kmeans.labels_

print("\n--- K-means (k=3) ---")
print(f"Cluster sizes: {df['KMeans_Cluster'].value_counts().sort_index().to_dict()}")
print(f"WCSS (inertia): {kmeans.inertia_:.4f}")


# 7. DBSCAN with eps=0.5, min_samples=3
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan.fit(df_scaled)
df['DBSCAN_Cluster'] = dbscan.labels_

n_clusters_db = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
n_noise_db = int((dbscan.labels_ == -1).sum())
print("\n--- DBSCAN (eps=0.5, min_samples=3) ---")
print(f"Found {n_clusters_db} cluster(s), {n_noise_db} noise point(s)")
print(f"Cluster sizes: {df['DBSCAN_Cluster'].value_counts().sort_index().to_dict()}")


# 6+8. Side-by-side visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Left: K-means
axes[0].scatter(
    df['AnnualIncome'], df['SpendingScore'],
    c=df['KMeans_Cluster'], cmap='viridis',
    s=60, edgecolor='k', alpha=0.85,
)
# Plot K-means centroids in original units
centers = scaler.inverse_transform(kmeans.cluster_centers_)
axes[0].scatter(
    centers[:, 0], centers[:, 1],
    marker='X', s=220, c='red', edgecolor='black', label='Centroids',
)
axes[0].set_title('K-Means (k=3) — every point gets a cluster')
axes[0].set_xlabel('Annual Income (in thousands)')
axes[0].set_ylabel('Spending Score (1-100)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: DBSCAN with noise highlighted
noise_mask = df['DBSCAN_Cluster'] == -1
axes[1].scatter(
    df.loc[~noise_mask, 'AnnualIncome'],
    df.loc[~noise_mask, 'SpendingScore'],
    c=df.loc[~noise_mask, 'DBSCAN_Cluster'], cmap='rainbow',
    s=60, edgecolor='k', alpha=0.85, label='Clustered',
)
axes[1].scatter(
    df.loc[noise_mask, 'AnnualIncome'],
    df.loc[noise_mask, 'SpendingScore'],
    c='black', marker='x', s=120, linewidths=3, label='Noise (-1)',
)
axes[1].set_title('DBSCAN (eps=0.5, min_samples=3) — density-based')
axes[1].set_xlabel('Annual Income (in thousands)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('K-Means vs DBSCAN on the same customer data', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('kmeans_vs_dbscan.png', dpi=120, bbox_inches='tight')
plt.show()


# Side-by-side comparison table
print("\n=== How the two algorithms classified each point ===")
comparison = df.groupby(['KMeans_Cluster', 'DBSCAN_Cluster']).size().unstack(fill_value=0)
print(comparison)


# Key takeaways
#
# - Same data, very different answers. K-means *partitions* — every point
#   has to land in some cluster, and with k=3 it splits the diagonal band
#   into two halves and gives the outliers their own group.
# - DBSCAN looks at *density*. The whole linear band is one dense region,
#   so it gets one label; the 3 outliers are far from everything else and
#   are labeled noise (-1).
# - Which is "right" depends on the question. For segmentation into a
#   fixed number of groups → K-means. For finding genuinely anomalous
#   customers and arbitrary-shaped clusters → DBSCAN.
# - Both require feature scaling. Both have a key parameter (k for
#   K-means, eps for DBSCAN) that you can't avoid tuning.
