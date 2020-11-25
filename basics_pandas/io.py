import pandas as pd

# reading data as a csv file using pandas
# bank_df = pd.read_csv("./")  # automatically converts it into dataframe class

# we can write a dataframe class as a csv file
portfolio_df = pd.DataFrame({
    'stock ticker symbol': ['AAPL', 'AMZN', 'T'],
    'number of stocks': [3, 4, 9],
    'price per share [$]': [3500, 200, 40]
})

# index is the side index for the rows
portfolio_df.to_csv('sample_output.csv', index=False)

# Reading tabular data from HTML
house_prices_df = pd.read_html(
    'https://www.livingin-canada.com/house-prices-canada.html')

# reading json
comments_df = pd.read_json("https://jsonplaceholder.typicode.com/comments")
print(comments_df)
