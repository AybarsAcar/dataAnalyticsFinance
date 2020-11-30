import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
# Principal Component Analysis
from sklearn.decomposition import PCA


# DATA INFO
# CUSTID: Identification of Credit Card holder
# BALANCE: Balance amount left in customer's account to make purchases
# BALANCE_FREQUENCY: How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)
# PURCHASES: Amount of purchases made from account
# ONEOFFPURCHASES: Maximum purchase amount done in one-go
# INSTALLMENTS_PURCHASES: Amount of purchase done in installment
# CASH_ADVANCE: Cash in advance given by the user
# PURCHASES_FREQUENCY: How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)
# ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)
# PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)
# CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid
# CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"
# PURCHASES_TRX: Number of purchase transactions made
# CREDIT_LIMIT: Limit of Credit Card for user
# PAYMENTS: Amount of Payment done by user
# MINIMUM_PAYMENTS: Minimum amount of payments made by user
# PRC_FULL_PAYMENT: Percent of full payment paid by user
# TENURE: Tenure of credit card service for user


# plot the correlations
def plot_correlations(df):
  correlations = df.corr()
  sns.heatmap(correlations, annot=True)
  plt.show()


def show_multiple_distplot(df):
  plt.figure(figsize=(10, 50))
  for i in range(n):
    plt.subplot(17, 1, i + 1)
    sns.distplot(df[df.columns[i]], kde_kws={'color': 'b', 'lw': 3, 'label': 'KDE'},
                 hist_kws={'color': 'g'})
    plt.title(df.columns[i])

  plt.tight_layout()
  plt.show()


# import the raw data
credit_card_df = pd.read_csv('../data/marketing_data.csv')

# exploration
print(credit_card_df.info())
print(credit_card_df.describe())

# OBSERVATIONS
# Mean balance is $1564
# Balance frequency is frequently updated on average ~0.9
# Purchases average is $1000
# one off purchase average is ~$600
# Average purchases frequency is around 0.5
# average ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, and CASH_ADVANCE_FREQUENCY are generally low
# Average credit limit ~ 4500
# Percent of full payment is 15%
# Average tenure is 11 years


most_spent = credit_card_df[credit_card_df['ONEOFF_PURCHASES'] == credit_card_df['ONEOFF_PURCHASES'].max()]
# print(most_spent)

max_cash_advance = credit_card_df['CASH_ADVANCE'].max()

# print(credit_card_df[credit_card_df['CASH_ADVANCE'] == max_cash_advance])

# explore missing data
# sns.heatmap(credit_card_df.isnull(), yticklabels=False,cbar=False, cmap='Blues')
# plt.show()

# print("NULL Values:")
# print(credit_card_df.isnull().sum())

# Fill the missing data
credit_card_df.loc[(credit_card_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = credit_card_df[
  'MINIMUM_PAYMENTS'].mean()

credit_card_df.loc[(credit_card_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = credit_card_df[
  'CREDIT_LIMIT'].mean()

# Customer id is not a valuable feature - so drop it in memory
# first check duplicated data
if credit_card_df.duplicated().sum() == 0:
  credit_card_df.drop('CUST_ID', axis=1, inplace=True)

# Kernel Density Estimate
n = len(credit_card_df.columns)  # number of columns, columns in an array, it is the array at index 0

# show_multiple_distplot(credit_card_df)

# plot_correlations(credit_card_df)


# Find the Optimal K using the Elbow Method
scaler = StandardScaler()
credit_card_df_scaled = scaler.fit_transform(credit_card_df)

# calc K-means for 20 K's
scores_1 = []
for i in range(1, 20):
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(credit_card_df_scaled)
  scores_1.append(kmeans.inertia_)

print(scores_1)
plt.plot(scores_1, 'bx-')
plt.title('WCSS vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# we choose the K to be 8
