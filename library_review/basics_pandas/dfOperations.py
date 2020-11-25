import pandas as pd

bank_client_df = pd.DataFrame({
    'Bank client ID': [111, 222, 333, 444],
    'Bank client Name': ['Chanel', 'Steve', 'Mitch', 'Ryan'],
    'Net Worth [$]': [3500, 29000, 10000, 2000],
    'Years with bank': [3, 4, 9, 5]
})

# we can pick certain rows that satisfy a certain criteria
# filter rows with clients who have more than 5 years
df_loyal = bank_client_df[(bank_client_df['Years with bank'] >= 5)]

# delete the bank client id column
# del bank_client_df['Bank client ID']

# select the inndividuals with high worth which are over 5,000 and sum their total worth
high_worth_cusomers = bank_client_df[(bank_client_df['Net Worth [$]'] >= 5000)]
total_worth = high_worth_cusomers['Net Worth [$]'].sum()
