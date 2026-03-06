# training data - 80%
# testing data - 20%

nums = [1,2,3,44,5,6,7,8]

"""
Height  Weight  Person
140     60      Abc
170     80      Def
150     65      Xyz

"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# import numpy as np

data = load_iris()

x = data.data # features (folder 1)
y = data.target # labels (folder 2)

# split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    train_size=0.8,
    random_state=40
)

print(x_train.shape)
print(x_test.shape)