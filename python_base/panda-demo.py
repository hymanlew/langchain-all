import pandas as pd

# pandas 主要用于数据分析，是基于 numpy 封装实现的
# 包含两个数据类型：Series（类似一维数组数据 + 其对应的索引），DataFrame（类似多维数组）

"""
1    10
2    10
3    10
dtype: int64
10 就是真正的数据（因此可看到数据类型为 int64），index表示数据的索引
"""
pd.Series(10, index=[1,2,3])

"""
a    1
b    2
c    3
dtype: int64
value 值是真正的数据（因此可看到数据类型为 int64），key 表示数据的索引
"""
pd.Series({'a':1,'b':2,'c':3})

"""
c    3.0
d    NaN
a    1.0
b    2.0
dtype: float64
value 值是数据，key 是数据的索引，并且指定了索引的顺序
"""
t = pd.Series({'a':1,'b':2,'c':3}, index=['c','a','b'])

# 使用两种方式的索引来取值
print(t['a'])
print(t[2])
print(t[['c','b','a']])

# 取指定范围内的索引的数据
print(t[:2])

# 只判断是否存在指定的索引，而不判断是否存在值
print('a' in t) # true
print(1 in t) # false
print(t.get('c', 10))

"""
9    0
8    1
7    2
6    3
5    4
dtype: int64
"""
import numpy as np
a = pd.Series(np.arange(5), index=np.arange(9,4,-1))
# 获取索引，获取数据
print(a.index)
print(a.values)

"""
a    NaN
b    NaN
c    NaN
d    4.0
e    NaN
f    NaN
dtype: float64
两个 Series 对齐计算，只会计算有相同索引的值，不相同的索引则为 NaN
"""
#
a = pd.Series([1,2,3], ['c','d','a'])
b = pd.Series([1,2,3,4], ['e','d','f','b'])
print(a + b)


"""
DataFrame 是一个表格型的数据类型，每列值类型可以不同。
既有行索引、也有列索引。常用于表达二维数据，但可以表达多维数据。

以下数据中，最外层的就是行索引，列索引
   0  1  2  3  4
0  0  1  2  3  4
1  5  6  7  8  9
"""
pd.DataFrame(np.arange(10).reshape((2,5)),)

dt = {
        'one': pd.Series([1,2,3], index=['a','b','c']),
        'two': pd.Series([9,8,7,6], index=["a",'b','c','d'])
    }
"""
   one  two
a  1.0    9
b  2.0    8
c  3.0    7
d  NaN    6
字典的 key 默认设为列索引，value 值中的索引作为行索引
"""
pd.DataFrame(dt)

"""
   two three
b    8   NaN
c    7   NaN
d    6   NaN
指定生成并输出对应 行索引，列索引的数据
"""
t = pd.DataFrame(dt, index=['b','c','d'], columns=['two','three'])


"""
.reindex，改变或重排 Series 和 DataFrame 索引
.reindex(index=None, columns=None, ...) 的参数

index,columns：指定新的行/列索引
fill_value：在重新索引时，用于填充缺失位置的值
method：填充方法，fill当前值向前填充，bfill向后填充
limit：最大填充量
copy：默认True，生成新的对象，False时，修改原对象
"""
t.reindex(columns=['two','one'])
print(t.index)
print(t.columns)

"""
.append(idx)：连接另一个Index对象，产生新的Index对象
.diff(idx)：计算差集，产生新的Index对象
.intersection(idx)：计算交集
.union(idx)：计算井集
.delete(loc)：删除loc位置处的元素
.insert(loc,e)：在loc位置增加一个元素e
.drop(idx)：删除 Series 和 DataFrame 指定的行或列索引（删除指定索引整组的数据）
"""
t.index.delete(0)
t.drop(['a', 'b'])

"""
     0      1
a    NaN    NaN
b    NaN    NaN
c    NaN    NaN
d    4.0    5.0
dtype: float64
两个 DataFrame 对齐计算，只会计算有相同索引的值，不相同的索引则为 NaN

.add(d,**argws)
.sub(d,**argws)
.mul(d,**argws)
.div(d,**argws)
"""
t.add(t, fill_value=0)
print(t - 1)
print(t == t)

"""
Series/DataFrame.sort_index(axis=0, ascending=True)：
在指定轴上根据索引进行排序（0 轴为行索引，1 轴为列索引），默认升序

Series.sort_values(axis=0, ascending=True)：
在指定轴上根据数值进行排序（0 轴为行索引，1 轴为列索引），默认升序

DataFrame.sort_values(by, axis=0, ascending=True)：
在指定轴 axis 上，指定的索引或索引列表 by，根据数值进行排序（0 轴为行索引，1 轴为列索引），默认升序

以上排序，NaN 值会统一放到排序的末尾（末尾行，末尾列）

Series/DataFrame.sum()：计算数据的总和，按0轴计算，下同
Series/DataFrame.count()：非NaN值的数量
Series/DataFrame.mean() .median()：计算数据的算术平均值、算术中位数
.var() .std()：计算数据的方差、标准差
Series/DataFrame.min() .max()：计算数据的最小值、最大值
"""

