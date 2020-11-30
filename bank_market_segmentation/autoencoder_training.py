import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, \
  AveragePooling2D, MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform  # Xavier Normal Initializer
from tensorflow.keras.optimizers import SGD

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

y_kmeans = kmeans.fit_predict(credit_card_df_scaled)

credit_card_df_cluster = pd.concat([credit_card_df, pd.DataFrame({'cluster': labels})], axis=1)

# 17 features as input
input_df = Input(shape=(17,))

# build encoder and decoder networks
# create a Dense Layer with 7 layers
x = Dense(7, activation='relu')(input_df)

# 500 neurons with glorot uniform initilisation
x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)
x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(x)

encoded = Dense(10, activation='relu', kernel_initializer='glorot_uniform')(x)

# build the decoder now
# it is the opposite of the encoder
x = Dense(2000, activation='relu', kernel_initializer='glorot_uniform')(encoded)
x = Dense(500, activation='relu', kernel_initializer='glorot_uniform')(x)

decoded = Dense(17, kernel_initializer='glorot_uniform')(x)

# Building the Entire Autoencoder
autoencoder = Model(input_df, decoded)

# Build the Encoder Network for the encoder section only
encoder = Model(input_df, encoded)

# Compile hte model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

print(credit_card_df_scaled.shape)

# TRAIN THE AUTOENCODER
# feed in the credit card scaled data to train the data
# it is an unsupervised training
autoencoder.fit(credit_card_df_scaled, credit_card_df_scaled, batch_size=128, epochs=25, verbose=1)
autoencoder.summary()

# from 17 features to 10 features
predictions = encoder.predict(credit_card_df_scaled)

# apply the KMeans algorithm based on the reduced 10 features ########################
# Optimal number of clusters
scores = []
for i in range(1, 20):
  # Apply KMeans, applied to the data with 10 features (reduced from 17)
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(predictions)
  scores.append(kmeans.inertia_)

# Plot the result
# plt.plot(scores, 'bx-')
# plt.title('Finding the optimal cluster number')
# plt.xlabel('Clusters')
# plt.ylabel('scores')
# plt.show()

# we can now select smaller number of clusters -> 4
kmeans = KMeans(4)
kmeans.fit(predictions)
labels = kmeans.labels_

# concat the cluster column to our Dataframe
df_cluster_dr = pd.concat([credit_card_df, pd.DataFrame({'cluster': labels})], axis=1)

# Principal Component Analysis
pca = PCA(n_components=2)
principal_component = pca.fit_transform(predictions)
pca_df = pd.DataFrame(data=principal_component, columns=['pca1', 'pca2'])

pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)

plt.figure(figsize=(10, 10))
ax = sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=pca_df, palette=['red', 'green', 'blue', 'yellow'])
plt.show()
