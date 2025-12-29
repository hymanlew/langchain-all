import pandas as pd
import numpy as np

dt = {
        'one': pd.Series([1,2,3], index=['a','b','c']),
        'two': pd.Series([9,8,7,6], index=["a",'b','c','d'])
    }
a = pd.DataFrame(dt)
print(a)

a = pd.DataFrame(dt, index=['b','c','d'], columns=['two','three'])
print(a)
