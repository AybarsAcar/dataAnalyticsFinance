import pandas as pd

bank_client_df = pd.DataFrame({
    'Bank client ID': [111, 222, 333, 444],
    'Bank client Name': ['Chanel', 'Steve', 'Mitch', 'Ryan'],
    'Net Worth [$]': [3500, 29000, 10000, 2000],
    'Years with bank': [3, 4, 9, 5]
})

# sort according to the numbers with the bank
# it doesnt sort them in memory though
bank_client_df.sort_values(by='Years with bank')
print(bank_client_df)

# we can force the changes in memory
bank_client_df.sort_values(by=['Years with bank'], inplace=True)
print(bank_client_df)
