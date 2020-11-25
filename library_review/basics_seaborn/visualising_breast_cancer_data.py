import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

# get the cancer data
# it is a classification type problem 0s and 1s
cancer = load_breast_cancer()

# turn the data into a DataFrame class
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(
    cancer['feature_names'], ['target']))

# see the first 5 rows and the bottom 5 rows
print(df_cancer.head())
print(df_cancer.tail())

# using seaborn to visualise the data
# plot the mean area and mean smoothness from the data frame
# hue -> colour code's the data points according to their target class [benign, malignant]
sns.scatterplot(x='mean area', y='mean smoothness',
                hue='target', data=df_cancer)

# prints out the counts as a bar graph based on the target
# sns.countplot(df_cancer['target'])

# plot the key features of the data
# and visualise the combinations of them
# vars -> takes an array of arguements we want to plot
sns.pairplot(df_cancer, hue='target', vars=[
             'mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])

# visualise the correlations in a heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(df_cancer.corr(), annot=True)

# distplot
sns.distplot(df_cancer['mean radius'], bins=25, color='blue')

#
# distplot for target == 1
class_1_df = df_cancer[df_cancer['target'] == 1]
# distplot for target == 0
class_0_df = df_cancer[df_cancer['target'] == 0]

# plot them seperately
plt.figure(figsize=(10, 7))
sns.distplot(class_0_df['mean radius'], bins=25, color='blue')
sns.distplot(class_1_df['mean radius'], bins=25, color='red')
plt.grid()
