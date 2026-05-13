"""
Practice activity: DBSCAN clustering on customer income/spending data.

Course: Microsoft Foundations of AI and Machine Learning
Module: AI & ML Algorithms and Techniques

DBSCAN's strength vs K-means: it finds clusters of arbitrary shape AND
labels outliers as noise (-1) without you having to pick k in advance.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# 3. Load the dataset
# 41 normal customers (income 15.0 -> 35.0 in 0.5 steps) plus 3 high-income
# outliers. SpendingScore follows a roughly linear trend with income for
# the normal points so they form one elongated dense band; the outliers
# sit far away on the income axis and should be flagged as noise.
annual_income = [
    15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
    20, 20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5,
    25, 25.5, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5,
    30, 30.5, 31, 31.5, 32, 32.5, 33, 33.5, 34, 34.5,
    35,
    80, 85, 90,  # outliers
]

spending_score = [
    # First 10 values match the activity's printed pattern (39, 42, ..., 66)
    39, 42, 45, 48, 51, 54, 57, 60, 63, 66,
    # Continue the upward trend, slowing down to stay within 1-100
    68, 70, 72, 73, 75, 76, 78, 79, 80, 82,
    83, 84, 85, 86, 87, 87, 88, 88, 89, 89,
    90, 91, 92, 93, 94, 95, 95, 96, 97, 98,
    99,
    # Outliers' spending scores
    40, 60, 80,
]

assert len(annual_income) == len(spending_score) == 44

df = pd.DataFrame({
    'AnnualIncome':  annual_income,
    'SpendingScore': spending_score,
})
print(df.head())
print(f"\nDataset shape: {df.shape}")


# 4. Preprocess: scale the features
scaler = StandardScaler()
df_scaled_array = scaler.fit_transform(df)
df_scaled = pd.DataFrame(df_scaled_array, columns=['AnnualIncome', 'SpendingScore'])
print("\nScaled data (head):")
print(df_scaled.head())


# 5. DBSCAN with the activity's default parameters
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan.fit(df_scaled)
df['Cluster'] = dbscan.labels_

n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
n_noise = int((dbscan.labels_ == -1).sum())
print(f"\n--- DBSCAN (eps=0.5, min_samples=3) ---")
print(f"Found {n_clusters} cluster(s), {n_noise} noise point(s)")
print(f"Cluster sizes: {df['Cluster'].value_counts().sort_index().to_dict()}")
print("\nNoise (outlier) rows:")
print(df[df['Cluster'] == -1])


# 6. Visualize the clusters
def plot_clusters(df, title, savepath):
    fig, ax = plt.subplots(figsize=(10, 6))
    noise_mask = df['Cluster'] == -1
    # Real clusters
    ax.scatter(
        df.loc[~noise_mask, 'AnnualIncome'],
        df.loc[~noise_mask, 'SpendingScore'],
        c=df.loc[~noise_mask, 'Cluster'], cmap='rainbow',
        s=60, edgecolor='k', alpha=0.85, label='Clustered',
    )
    # Noise points styled distinctly
    if noise_mask.any():
        ax.scatter(
            df.loc[noise_mask, 'AnnualIncome'],
            df.loc[noise_mask, 'SpendingScore'],
            c='black', marker='x', s=120, linewidths=3, label='Noise (-1)',
        )
    ax.set_title(title)
    ax.set_xlabel('Annual Income (in thousands)')
    ax.set_ylabel('Spending Score (1-100)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(savepath, dpi=120, bbox_inches='tight')
    plt.show()


plot_clusters(df, 'DBSCAN Clustering of Customers (eps=0.5, min_samples=3)',
              'dbscan_eps_0_5.png')


# 7. Tune the parameters — sweep eps and min_samples
print("\n--- Parameter sweep ---")
print(f"{'eps':>6s}  {'min_samples':>11s}  {'clusters':>9s}  {'noise pts':>10s}")
for eps_val in [0.3, 0.5, 0.7, 1.0]:
    for ms in [3, 5]:
        m = DBSCAN(eps=eps_val, min_samples=ms).fit(df_scaled)
        nc = len(set(m.labels_)) - (1 if -1 in m.labels_ else 0)
        nn = int((m.labels_ == -1).sum())
        print(f"{eps_val:>6.2f}  {ms:>11d}  {nc:>9d}  {nn:>10d}")


# Re-run with eps=0.7 to compare with the activity's suggested second pass
dbscan_07 = DBSCAN(eps=0.7, min_samples=3).fit(df_scaled)
df['Cluster'] = dbscan_07.labels_

n_clusters_07 = len(set(dbscan_07.labels_)) - (1 if -1 in dbscan_07.labels_ else 0)
n_noise_07 = int((dbscan_07.labels_ == -1).sum())
print(f"\n--- DBSCAN (eps=0.7, min_samples=3) ---")
print(f"Found {n_clusters_07} cluster(s), {n_noise_07} noise point(s)")

plot_clusters(df, 'DBSCAN Clustering of Customers (eps=0.7, min_samples=3)',
              'dbscan_eps_0_7.png')


# Key takeaways
#
# - DBSCAN groups points by *density*: a core point has at least
#   `min_samples` neighbors within distance `eps`. Points reachable from
#   core points join the cluster. Everything else is labeled -1 (noise).
# - You don't pick k. Clusters emerge from the density structure of the
#   data — that's why DBSCAN can find arbitrarily-shaped clusters that
#   K-means misses (e.g. crescents, rings).
# - Scaling matters. DBSCAN uses Euclidean distance, so a feature with a
#   larger numeric range otherwise dominates the eps neighborhood.
# - eps is the dial that matters most. Too small → many noise points;
#   too large → distinct clusters merge. The "k-distance plot" (sorted
#   distances to the kth nearest neighbor) is the standard way to pick a
#   sensible eps; on this dataset eps≈0.5 cleanly isolates the 3 outliers.
