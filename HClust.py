import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load and prepare the data
data = pd.read_csv('/Users/tadeozuniga/PycharmProjects/508-final/data/Alasak_cleaned.csv')
features = ['elev_m', 'shrub', 'longitude', 'latitude', 'dshrub', 'point']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Hierarchical clustering
Z = linkage(X_scaled, method='ward')

# Plotting a truncated dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z, truncate_mode='lastp', p=12, show_leaf_counts=True, leaf_rotation=90, leaf_font_size=12, show_contracted=True)
plt.title('Truncated Hierarchical Clustering Dendrogram')
plt.xlabel('Cluster size')
plt.ylabel('Distance')
plt.axhline(y=7, color='r', linestyle='--')
plt.show()

# Determining the number of clusters
max_d = 7
clusters = fcluster(Z, max_d, criterion='distance')
data['Cluster'] = clusters

# Heatmap with Clustering
sns.clustermap(data[features], standard_scale=1, method='ward', cmap='viridis')
plt.show()
