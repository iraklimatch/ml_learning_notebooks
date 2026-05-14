"""
Practice activity: Combining unsupervised learning methods.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques

Run K-means, DBSCAN, and PCA on the same 10-customer / 3-feature
dataset and compare the cluster assignments and the reduced-space
layout. The activity's snippets accidentally append cluster labels
to `df_scaled` and then re-use that DataFrame for DBSCAN and PCA;
we keep labels in a separate DataFrame so the dimensionality
reduction sees only the original features.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# 3. Load the dataset (10 customers, 3 features)
data = {
    'AnnualIncome':  [15, 16, 17, 18, 19, 20, 22, 25, 30, 35],
    'SpendingScore': [39, 81,  6, 77, 40, 76, 94,  5, 82, 56],
    'Age':           [20, 22, 25, 24, 35, 40, 30, 21, 50, 31],
}
df = pd.DataFrame(data)
print(df)


# 4. Preprocess: scale all features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(features_scaled, columns=df.columns)
print("\nScaled features:")
print(df_scaled)


# Keep cluster labels in a separate frame so the dim-reduction
# step doesn't accidentally see them as features.
results = pd.DataFrame(index=df.index)


# 5. Task 1 — K-means (k=3)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
results['KMeans_Cluster'] = kmeans.fit_predict(features_scaled)
print("\n--- K-means (k=3) ---")
print(f"Cluster sizes: {results['KMeans_Cluster'].value_counts().sort_index().to_dict()}")
print(f"WCSS (inertia): {kmeans.inertia_:.4f}")


# 6. Task 2 — DBSCAN (eps=0.5, min_samples=2 as specified by the activity)
dbscan = DBSCAN(eps=0.5, min_samples=2)
results['DBSCAN_Cluster'] = dbscan.fit_predict(features_scaled)
n_clusters_db = len(set(results['DBSCAN_Cluster'])) - (1 if -1 in results['DBSCAN_Cluster'].values else 0)
n_noise_db = int((results['DBSCAN_Cluster'] == -1).sum())
print(f"\n--- DBSCAN (eps=0.5, min_samples=2) ---")
print(f"Found {n_clusters_db} cluster(s), {n_noise_db} noise point(s)")
print(f"Cluster sizes: {results['DBSCAN_Cluster'].value_counts().sort_index().to_dict()}")


# 7. Task 3 — PCA (3D -> 2D)
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(features_scaled)
df_pca = pd.DataFrame(pca_coords, columns=['PCA1', 'PCA2'])
print(f"\n--- PCA ---")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative variance:      {pca.explained_variance_ratio_.cumsum()}")
print("Loadings (rows=PC, cols=feature):")
print(pd.DataFrame(pca.components_, columns=df.columns, index=['PC1', 'PC2']).round(3))


# 8. Visualization — three panels: K-means, DBSCAN, PCA-with-K-means
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

# K-means in original space
axes[0].scatter(
    df['AnnualIncome'], df['SpendingScore'],
    c=results['KMeans_Cluster'], cmap='viridis',
    s=140, edgecolor='k', alpha=0.9,
)
axes[0].set_title('K-Means clusters (k=3) in original space')
axes[0].set_xlabel('Annual Income (in thousands)')
axes[0].set_ylabel('Spending Score (1-100)')
axes[0].grid(True, alpha=0.3)

# DBSCAN in original space — noise highlighted with black X
noise_mask = results['DBSCAN_Cluster'] == -1
axes[1].scatter(
    df.loc[~noise_mask, 'AnnualIncome'], df.loc[~noise_mask, 'SpendingScore'],
    c=results.loc[~noise_mask, 'DBSCAN_Cluster'], cmap='rainbow',
    s=140, edgecolor='k', alpha=0.9, label='Clustered',
)
if noise_mask.any():
    axes[1].scatter(
        df.loc[noise_mask, 'AnnualIncome'], df.loc[noise_mask, 'SpendingScore'],
        c='black', marker='x', s=160, linewidths=3, label='Noise (-1)',
    )
axes[1].set_title(f'DBSCAN (eps=0.5, min_samples=2) — {n_noise_db} noise pt(s)')
axes[1].set_xlabel('Annual Income (in thousands)')
axes[1].set_ylabel('Spending Score (1-100)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# PCA, colored by K-means cluster
axes[2].scatter(
    df_pca['PCA1'], df_pca['PCA2'],
    c=results['KMeans_Cluster'], cmap='viridis',
    s=140, edgecolor='k', alpha=0.9,
)
axes[2].set_title(f'PCA 2D ({pca.explained_variance_ratio_.sum()*100:.1f}% variance), colored by K-means')
axes[2].set_xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[2].set_ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[2].grid(True, alpha=0.3)

plt.suptitle('K-Means vs DBSCAN vs PCA on the same customer data', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('three_methods.png', dpi=120, bbox_inches='tight')
plt.show()


# 9. Interpretation aids — cluster profiles in original units
df_full = pd.concat([df, results], axis=1)
print("\n=== K-means cluster profiles (mean per cluster, original units) ===")
print(df_full.groupby('KMeans_Cluster')[['AnnualIncome', 'SpendingScore', 'Age']].mean().round(2))

print("\n=== DBSCAN cluster profiles (mean per cluster, original units) ===")
print(df_full.groupby('DBSCAN_Cluster')[['AnnualIncome', 'SpendingScore', 'Age']].mean().round(2))

print("\n=== How the two clusterings agreed/disagreed ===")
print(pd.crosstab(results['KMeans_Cluster'], results['DBSCAN_Cluster'],
                  rownames=['KMeans'], colnames=['DBSCAN']))


# Reflection
#
# - K-means vs DBSCAN. K-means forced all 10 customers into 3 groups
#   based on proximity to centroids. DBSCAN at eps=0.5/min_samples=2
#   was more selective: it only formed clusters where points were dense
#   enough, and labeled everything else as noise. With this small a
#   sample the two methods can disagree quite a lot.
#
# - Outliers. DBSCAN's -1 label is exactly the kind of behavior you
#   want for anomaly-style problems — instead of pretending an isolated
#   point belongs somewhere, DBSCAN refuses to assign it. K-means has
#   no equivalent and would have put those outliers into whichever
#   centroid was nearest.
#
# - PCA. Two principal components capture about 87% of the variance.
#   PC1 picks up overall scale (positive loadings on all three features),
#   PC2 contrasts Spending Score against Annual Income.
#
# - Reduced vs original. The K-means clusters that look distinct in the
#   Income x Spending plot also separate in the PCA plot, which is a
#   useful sanity check — the clustering is finding real structure in
#   the data, not just slicing along one feature axis.
