import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

# Load and prepare data
data_path = '/Users/tadeozuniga/PycharmProjects/508-final/data/Alasak_cleaned.csv'
data = pd.read_csv(data_path)
top_features = ['elev_m', 'shrub', 'longitude', 'latitude', 'dshrub', 'point']
X = data[top_features]

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[['elev_m', 'shrub', 'dshrub', 'point']])  # Scale non-geographic features

# Hierarchical clustering
cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0, compute_full_tree=True)
data['Cluster'] = cluster.fit_predict(X_scaled)

# Prepare a Basemap for Alaska focused on the southwest with a broader view
fig, ax = plt.subplots(figsize=(10, 10))
m = Basemap(resolution='i',  # intermediate resolution
            projection='merc',  # Mercator projection
            lat_0=57.0, lon_0=-157.0,  # Adjusted center for a broader view
            llcrnrlon=-170, llcrnrlat=52,  # Adjusted lower left corner
            urcrnrlon=-142, urcrnrlat=62)  # Adjusted upper right corner

# Draw coastlines, states, and countries
m.drawcoastlines()
m.drawstates()
m.drawcountries()

# Draw a physical map background
m.shadedrelief()  # Adds a nice physical detail

# Plot each cluster with different colors
for cluster in data['Cluster'].unique():
    subset = data[data['Cluster'] == cluster]
    x, y = m(subset['longitude'].values, subset['latitude'].values)
    m.scatter(x, y, s=30, label=f'Cluster {cluster}', alpha=0.6, edgecolors='none')

# Add a legend and a title
plt.legend(loc='upper right')
plt.title('Spatial Distribution of Bird Species Clusters in Southwest Alaska')

# Save the plot as a PNG file
plt.savefig('/Users/tadeozuniga/PycharmProjects/508-final/Southwest_Alaska_clusters.png', format='png', dpi=300)
plt.close()

print("Map with a broader focus on Southwest Alaska has been saved.")
