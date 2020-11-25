import pandas as pd

df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])

# concat the dataframes
df_combined = pd.concat([df1, df2, df3])
print(df_combined)

#
raw_data = {
    'Bank Client ID': ['1', '2', '3', '4', '5'],
    'First Name': ['Aybars', 'Isil', 'Zuhre', 'Shuang', 'Artimis'],
    'Last Name': ['Acar', 'Sozgec', 'Acar', 'Acar', "Zhao"]
}

bank_df_1 = pd.DataFrame(
    raw_data, columns=['Bank Client ID', 'First Name', 'Last Name'])
print(bank_df_1)

raw_data_2 = {
    'Bank Client ID': ['6', '7', '8', '9', '10'],
    'First Name': ['Bill', 'Sarah', 'Ali', 'Mohammad', 'LeBron'],
    'Last Name': ['George', 'Jay', 'Sozgec', 'Ali', "James"]
}

bank_df_2 = pd.DataFrame(
    raw_data_2, columns=['Bank Client ID', 'First Name', 'Last Name'])

raw_data_3 = {
    'Bank Client ID': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'Annual Salary [$/yr]': [250000, 123420, 1041, 14001, 1234, 134910, 1398301, 1301, 30902, 139041]
}

bank_df_salary = pd.DataFrame(
    raw_data_3, columns=['Bank Client ID', 'Annual Salary [$/yr]'])

# combine them
bank_df_all = pd.concat([bank_df_1, bank_df_2])
print()
print(bank_df_all)

bank_full_df = pd.merge(bank_df_all, bank_df_salary, on='Bank Client ID')
print()
print(bank_full_df)

# Add a new customer to the df
new_client_df = pd.DataFrame({
    'Bank Client ID': ['11'],
    'First Name': ['Michael'],
    'Last Name': ['Jordan'],
    'Annual Salary [$/yr]': [1394020]
}, columns=[
    'Bank Client ID', 'First Name', 'Last Name', 'Annual Salary [$/yr]'])

print()
print(pd.concat([bank_full_df, new_client_df]))
