import pandas as pd

data = pd.read_csv("coffee_shop_sales.csv")

# print(data.head(10))
# print(data.tail())

# print(data.info())
print(data.describe())