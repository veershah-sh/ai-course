import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("coffee_shop_sales.csv")

# 1. add total_amount column 

data['total_amount'] = data['unit_price'] * data['transaction_qty']

# 2. find total sales

# total_sales = data['total_amount'].sum()

# print(f"Total Sales: {total_sales}")

# 1. total sales by product category
product_category = data.groupby('product_category')['total_amount'].sum()
plt.bar(product_category.index, product_category.values)

plt.xlabel("Product Category")
plt.ylabel("Total Sales")

plt.title("Total Sales by Product Category")

plt.show()
