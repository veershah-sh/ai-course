import numpy as np
import datetime as dt

arr = np.array([1,2,3,4,5])

print(dt.time())
for a in arr:
    print(a+2, end=" ")
print(dt.time())

print(arr+2)
print(np.sqrt(arr))
