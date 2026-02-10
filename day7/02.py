# 2d dataframe

import pandas as pd

data = {
    "name": ["abc", "xyz", "def"],
    "rollno": [1,2,3],
    "isPresent": [True, False, True]
}

df = pd.DataFrame(data)

print(df)