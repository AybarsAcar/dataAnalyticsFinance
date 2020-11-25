import pandas as pd

bank_client_df = pd.DataFrame({
    'Bank client ID': [111, 222, 333, 444],
    'Bank client Name': ['Chanel', 'Steve', 'Mitch', 'Ryan'],
    'Net Worth [$]': [3500, 29000, 10000, 2000],
    'Years with bank': [3, 4, 9, 5]
})

# define a function that increases all clients networth with a fixed value


def networth_update(balance):
    return balance * (1.1)


# apply the function to the dataframe class
bank_client_df['Net Worth [$]'].apply(networth_update)

print(bank_client_df)

# get the length of the client names
bank_client_df['Bank client Name'].apply(len)

# sum up the years
bank_client_df['Years with bank'].sum()
