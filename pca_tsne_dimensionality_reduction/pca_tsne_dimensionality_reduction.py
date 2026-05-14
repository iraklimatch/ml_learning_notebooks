"""
Practice activity: Dimensionality reduction with PCA and t-SNE.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques

Reduce 3 features (AnnualIncome, SpendingScore, Age) -> 2 dimensions
using PCA (linear, variance-preserving) and t-SNE (non-linear,
local-structure-preserving), and compare the resulting layouts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# 3. Load the dataset (3 features now: Income, Spending, Age)
annual_income = [
    15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
    20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
    25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
    30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
    35,
    80, 85, 90,  # outliers
]

# SpendingScore: same construction as the DBSCAN activity — a smooth
# upward trend through the 41 normal customers, then 3 outlier values.
spending_score = [
    39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
    68, 70, 72, 73, 75, 76, 78, 79, 80, 82,
    83, 84, 85, 86, 87, 87, 88, 88, 89, 89,
    90, 91, 92, 93, 94, 95, 95, 96, 97, 98,
    99,
    40, 60, 80,
]

# Age: roughly increasing with income for the normal customers (so PCA
# has a real correlation to compress), then 3 outliers in their 50s-60s.
age = [
    22, 25, 23, 28, 30, 26, 24, 29, 27, 31,
    33, 29, 32, 35, 34, 28, 30, 27, 32, 36,
    38, 42, 40, 45, 47, 44, 49, 46, 50, 48,
    52, 55, 53, 51, 56, 54, 58, 57, 59, 60,
    62,
    55, 60, 65,
]

assert len(annual_income) == len(spending_score) == len(age) == 44

df = pd.DataFrame({
    'AnnualIncome':  annual_income,
    'SpendingScore': spending_score,
    'Age':           age,
})
print(df.head())
print(f"\nDataset shape: {df.shape}")
print("\nFeature correlations (before scaling):")
print(df.corr().round(3))


# 4. Preprocess: scale all three features
scaler = StandardScaler()
scaled = scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled, columns=['AnnualIncome', 'SpendingScore', 'Age'])
print("\nScaled data (head):")
print(df_scaled.head())


# Tag normal vs outlier so we can color the reduced plots consistently
is_outlier = np.array([False] * 41 + [True] * 3)
colors = np.where(is_outlier, 'red', 'steelblue')


# 5. PCA: 3 -> 2
pca = PCA(n_components=2)
df_pca = pca.fit_transform(scaled)
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])
print("\n--- PCA results ---")
print(df_pca.head())
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative variance:      {pca.explained_variance_ratio_.cumsum()}")
print(f"PC loadings (rows=PC, cols=feature):")
print(pd.DataFrame(pca.components_,
                   columns=df.columns,
                   index=['PC1', 'PC2']).round(3))


# 6. Visualize PCA
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df_pca['PCA1'], df_pca['PCA2'], c=colors, s=70, edgecolor='k', alpha=0.85)
# Tag outliers
for i, is_o in enumerate(is_outlier):
    if is_o:
        ax.annotate(f"income={df.iloc[i]['AnnualIncome']:.0f}",
                    (df_pca.iloc[i]['PCA1'], df_pca.iloc[i]['PCA2']),
                    xytext=(8, 4), textcoords='offset points', fontsize=9)
ax.set_title(f'PCA — 3D -> 2D '
             f'({pca.explained_variance_ratio_.sum()*100:.1f}% variance captured)')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_result.png', dpi=120, bbox_inches='tight')
plt.show()


# 7. t-SNE: 3 -> 2
# perplexity=3 matches the activity; small perplexity emphasizes local
# neighborhoods, which is appropriate for a 44-row dataset.
tsne = TSNE(n_components=2, perplexity=3, random_state=42, init='pca')
df_tsne = tsne.fit_transform(scaled)
df_tsne = pd.DataFrame(df_tsne, columns=['t-SNE1', 't-SNE2'])
print("\n--- t-SNE results ---")
print(df_tsne.head())


# 8. Visualize t-SNE
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(df_tsne['t-SNE1'], df_tsne['t-SNE2'], c=colors, s=70, edgecolor='k', alpha=0.85)
for i, is_o in enumerate(is_outlier):
    if is_o:
        ax.annotate(f"income={df.iloc[i]['AnnualIncome']:.0f}",
                    (df_tsne.iloc[i]['t-SNE1'], df_tsne.iloc[i]['t-SNE2']),
                    xytext=(8, 4), textcoords='offset points', fontsize=9)
ax.set_title('t-SNE — 3D -> 2D (perplexity=3)')
ax.set_xlabel('t-SNE1')
ax.set_ylabel('t-SNE2')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_result.png', dpi=120, bbox_inches='tight')
plt.show()


# Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(df_pca['PCA1'], df_pca['PCA2'], c=colors, s=70, edgecolor='k', alpha=0.85)
axes[0].set_title(f'PCA (linear) — {pca.explained_variance_ratio_.sum()*100:.1f}% variance')
axes[0].set_xlabel('PCA1')
axes[0].set_ylabel('PCA2')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(df_tsne['t-SNE1'], df_tsne['t-SNE2'], c=colors, s=70, edgecolor='k', alpha=0.85)
axes[1].set_title('t-SNE (non-linear, perplexity=3)')
axes[1].set_xlabel('t-SNE1')
axes[1].set_ylabel('t-SNE2')
axes[1].grid(True, alpha=0.3)

plt.suptitle('PCA vs t-SNE on the same 3-feature customer data '
             '(red = outliers)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('pca_vs_tsne.png', dpi=120, bbox_inches='tight')
plt.show()


# Key takeaways
#
# - PCA is a *linear* projection. PC1 is the direction of maximum variance
#   in the data; PC2 is the next-best orthogonal direction. The loadings
#   tell you which original features each PC is built from. PCA preserves
#   global structure: distances on the PCA plot roughly match distances
#   in the original space.
#
# - t-SNE is *non-linear* and only preserves *local* structure. Points
#   that are neighbors in the original space stay neighbors; points that
#   were far apart can end up anywhere. The cluster *shapes* and the
#   *distances between clusters* on a t-SNE plot are not meaningful —
#   don't read them as you would PCA.
#
# - Perplexity controls how "local" t-SNE is. Small perplexity (3-5) on
#   small datasets emphasizes near neighbors. Rule of thumb: perplexity
#   between 5 and 50, and never larger than n-1.
#
# - When to use which:
#     PCA  — first pass on tabular data, feature compression for a
#            downstream model, anywhere you need a deterministic and
#            invertible transform.
#     t-SNE — exploratory visualization of high-dimensional data (images,
#            embeddings); great for *seeing* clusters that already exist,
#            not for measuring them. UMAP is a faster modern alternative.
