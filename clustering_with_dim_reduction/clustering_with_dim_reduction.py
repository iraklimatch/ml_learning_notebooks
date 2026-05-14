"""
Practice activity: Combining K-means with PCA / t-SNE.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques

Run K-means on the original 3-feature space, then reduce to 2D with
PCA *and* t-SNE so the same cluster labels can be compared in each
projection.

Note: the activity's snippets accidentally include the new
`KMeans_Cluster` column in the data passed to PCA/t-SNE. We keep that
column separate so the dimensionality reduction sees only the original
features.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# 3. Load the dataset (10 customers, 3 features)
data = {
    'AnnualIncome':  [15, 16, 17, 18, 19, 20, 22, 25, 30, 35],
    'SpendingScore': [39, 81,  6, 77, 40, 76, 94,  5, 82, 56],
    'Age':           [20, 22, 25, 24, 35, 40, 30, 21, 50, 31],
}
df = pd.DataFrame(data)
print(df)


# 4. Preprocess — scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(features_scaled, columns=df.columns)
print("\nScaled data:")
print(df_scaled)


# 5. K-means clustering on the scaled features
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(df_scaled)
df['KMeans_Cluster'] = labels
print("\n--- K-means (k=3) ---")
print(f"Cluster sizes: {df['KMeans_Cluster'].value_counts().sort_index().to_dict()}")
print(f"WCSS (inertia): {kmeans.inertia_:.4f}")
print(df)


# 6. Visualize K-means clusters in the original feature space
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(
    df['AnnualIncome'], df['SpendingScore'],
    c=df['KMeans_Cluster'], cmap='viridis',
    s=120, edgecolor='k', alpha=0.9,
)
ax.set_title('K-Means Clustering of Customers (Income vs Spending)')
ax.set_xlabel('Annual Income (in thousands)')
ax.set_ylabel('Spending Score (1-100)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_original_space.png', dpi=120, bbox_inches='tight')
plt.show()


# 7. PCA — reduce 3D -> 2D
pca = PCA(n_components=2)
df_pca = pca.fit_transform(features_scaled)
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print("\n--- PCA ---")
print(df_pca)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative variance:      {pca.explained_variance_ratio_.cumsum()}")
print("Loadings (rows=PC, cols=feature):")
print(pd.DataFrame(pca.components_, columns=df.columns[:3], index=['PC1', 'PC2']).round(3))


# 7b. t-SNE — reduce 3D -> 2D
# perplexity must be < n_samples; we use 5 as suggested by the activity.
tsne = TSNE(n_components=2, perplexity=5, random_state=42, init='pca')
df_tsne = tsne.fit_transform(features_scaled)
df_tsne = pd.DataFrame(df_tsne, columns=['t-SNE1', 't-SNE2'])
print("\n--- t-SNE (perplexity=5) ---")
print(df_tsne)


# 8. Visualize both projections side by side, colored by K-means cluster
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(df_pca['PCA1'], df_pca['PCA2'],
                c=df['KMeans_Cluster'], cmap='viridis',
                s=140, edgecolor='k', alpha=0.9)
axes[0].set_title(f'PCA — colored by K-means cluster '
                  f'({pca.explained_variance_ratio_.sum()*100:.1f}% variance)')
axes[0].set_xlabel(f'PCA1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PCA2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df_tsne['t-SNE1'], df_tsne['t-SNE2'],
                c=df['KMeans_Cluster'], cmap='viridis',
                s=140, edgecolor='k', alpha=0.9)
axes[1].set_title('t-SNE — colored by K-means cluster (perplexity=5)')
axes[1].set_xlabel('t-SNE1')
axes[1].set_ylabel('t-SNE2')
axes[1].grid(True, alpha=0.3)

plt.suptitle('K-Means clusters visualized via PCA vs t-SNE', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('pca_vs_tsne_with_clusters.png', dpi=120, bbox_inches='tight')
plt.show()


# 9. Cluster profiles in original units (for interpretation)
print("\n=== Cluster profiles (means per cluster, original units) ===")
print(df.groupby('KMeans_Cluster')[['AnnualIncome', 'SpendingScore', 'Age']].mean().round(2))


# Reflection
#
# - K-means grouping. On the 3-feature scaled data, K-means produced three
#   clearly distinct clusters. Looking at the means by cluster makes the
#   interpretation easy: one cluster is the very-low-spending customers,
#   one is the high-spending mid-income customers, and one is the older
#   high-income customer(s).
#
# - PCA. With only 3 features and moderate correlation, the two principal
#   components capture the bulk of the variance — see the printed
#   "Cumulative variance" line. The clusters separate visibly in the PCA
#   plot, which means the structure K-means found is also visible in the
#   directions of largest variance.
#
# - t-SNE. Preserves *local* neighborhoods, so points that K-means
#   considered close are placed close in the t-SNE plot too. Inter-cluster
#   distance on the t-SNE plot is not meaningful — you can't say cluster A
#   is "closer to" cluster B than to cluster C just from the t-SNE layout.
#
# - Takeaway. Running clustering in original space and then *coloring* the
#   reduced-space plot by cluster is a common QA loop: if a clean cluster
#   structure also appears separated in the PCA / t-SNE plot, the
#   clustering is finding real geometric structure in the data — not
#   artefacts of one specific axis.
