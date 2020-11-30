import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
# Principal Component Analysis
from sklearn.decomposition import PCA

# import the raw data
credit_card_df = pd.read_csv('../data/marketing_data.csv')

# Fill the missing data
credit_card_df.loc[(credit_card_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = credit_card_df[
  'MINIMUM_PAYMENTS'].mean()

credit_card_df.loc[(credit_card_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = credit_card_df[
  'CREDIT_LIMIT'].mean()

# Customer id is not a valuable feature - so drop it in memory
# first check duplicated data
if credit_card_df.duplicated().sum() == 0:
  credit_card_df.drop('CUST_ID', axis=1, inplace=True)

# Find the Optimal K using the Elbow Method
scaler = StandardScaler()
credit_card_df_scaled = scaler.fit_transform(credit_card_df)

# Applying K-means for K=8
kmeans = KMeans(8)
kmeans.fit(credit_card_df_scaled)
labels = kmeans.labels_

cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=[credit_card_df.columns])
print(cluster_centers)

y_kmeans = kmeans.fit_predict(credit_card_df_scaled)
print(y_kmeans)

credit_card_df_cluster = pd.concat([credit_card_df, pd.DataFrame({'cluster': labels})], axis=1)
print(credit_card_df_cluster.head())

# plot the histogram for clusters
# for i in credit_card_df.columns:
#   plt.figure(figsize=(35, 5))
#   for j in range(8):
#     plt.subplot(1, 8, j + 1)
#     cluster = credit_card_df_cluster[credit_card_df_cluster['cluster'] == j]
#     cluster[i].hist(bins=20)
#     plt.title("{}\n Cluster{}".format(i, j))
#
# plt.show()

# Obtain the principal components -> compress it to 2 components
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(credit_card_df_scaled)
# print(principal_comp)

# convert it into a DataFrame
pca_df = pd.DataFrame(data=principal_comp, columns=['pca1', 'pca2'])
# print(pca_df)

# Concatenate the clusters labels to the dataframe
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)
print(pca_df)

# PLot it
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=pca_df,
                     palette=['red', 'green', 'blue', 'pink', 'yellow', 'gray', 'purple', 'black'])
plt.show()