import pandas as pd

data = pd.read_csv("coffee_shop_sales.csv")

# print(data.head(10))
# print(data.tail())

# print(data.info())
# print(data.describe())


# print(data["store_location"])
# print(data[["unit_price", "product_category"]].head(10))

# print(data.iloc[1050])
# data.loc['unit_price'] > 3.0
print(data.loc[data["unit_price"] > 3.0])

# check missing data
print(data.isna())

# replace missing data
data.fillna(data["unit_price"].min(), inplace=True)
