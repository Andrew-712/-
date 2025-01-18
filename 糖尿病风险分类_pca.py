import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('diabetes_data.csv')
print(data.head())

#pca主成分分析 可视化 聚类  将若干特征投影到二维平面
# Preprocess data: Select numerical columns for clustering and PCA analysis
numerical_cols = ['weight', 'height', 'blood_glucose', 'physical_activity',
                  'sleep_hours', 'bmi', 'risk_score']

# Standardize the numerical data for clustering and PCA
scaler = StandardScaler()
sd_data = scaler.fit_transform(data[numerical_cols])

# Perform PCA analysis
pca = PCA(n_components=2)
pca_result = pca.fit_transform(sd_data)
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

# Add PCA components to the dataset for visualization
data['PC1'] = pca_df['PC1']
data['PC2'] = pca_df['PC2']

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(sd_data)
data['Cluster'] = clusters

# Visualize PCA and clusters
plt.figure(figsize=(10, 6))
for cluster in np.unique(clusters):
    plt.scatter(
        data.loc[data['Cluster'] == cluster, 'PC1'],
        data.loc[data['Cluster'] == cluster, 'PC2'],
        label=f'Cluster {cluster}'
    )

plt.title('PCA and Clustering Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid()
plt.show()
#label:cluster0-低风险 cluster1-中风险 cluster2-高风险

# Explain variance captured by PCA
explained_variance = pca.explained_variance_ratio_
explained_variance_cumsum = np.cumsum(explained_variance)

print(explained_variance, explained_variance_cumsum)


