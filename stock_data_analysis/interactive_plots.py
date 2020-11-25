import pandas as pd
import plotly.express as px

# read the stocks data
stocks_df = pd.read_csv("../data/stocks.csv")
# sort by date
stocks_df = stocks_df.sort_values(by=['Date'])


# interactive plotting using plotly
# @input df -> DataFrame, title -> String
def interactive_plot(df, title):
  fig = px.line(title=title)

  # add scatter trace to it
  for i in df.columns[1:]:
    fig.add_scatter(x=df['Date'], y=df[i], name=title)
  fig.show()


# interactive_plot(stocks_df, "Stocks")

def normalise(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i] / x[i][0]
  return x


interactive_plot(normalise(stocks_df), "Stocks Normalised")
