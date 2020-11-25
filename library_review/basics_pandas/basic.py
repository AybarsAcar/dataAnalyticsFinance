import pandas as pd

# define a list
my_list = ["AAPL", "AMZN", "T"]
label = ['stock#1', 'stock#2', 'stock#3']

# create a 1 dimensional Pandas "series"
# series is a single row of data
# they are associated with their labels
x_series = pd.Series(data=my_list, index=label)
print(x_series)
print(type(x_series))

# create pandas data frame
# you define it from a Dictionary
bank_client_df = pd.DataFrame({
    'Bank client ID': [111, 222, 333, 444],
    'Bank client Name': ['Chanel', 'Steve', 'Mitch', 'Ryan'],
    'Net Worth [$]': [3500, 29000, 10000, 2000],
    'Years with bank': [3, 4, 9, 5]
})

print(bank_client_df)

# get the first couple of rows
bank_client_df.head()
# first 2 rows only
bank_client_df.head(2)

# get the last couple of elements
bank_client_df.tail()
# last row only
bank_client_df.tail(1)

print()
# create a portfolio data frame
portfolio_df = pd.DataFrame({
    'stock ticker symbol': ['AAPL', 'AMZN', 'T'],
    'number of stocks': [3, 4, 9],
    'price per share [$]': [3500, 200, 40]
})
print(portfolio_df)

portfolio_value = portfolio_df['number of stocks'] * \
    portfolio_df['price per share [$]']
print('Total portfolio value = ' + str(portfolio_value.sum()))
