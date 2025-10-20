import numpy as np
import matplotlib.pyplot as plt

a = np.arange(100).reshape(5, 20)

"""
注意：np.savetxt，np.loadtxt 只能存储/读取 一维/二维的数据
"""
# frame: 字符串，表示文件或产生器的名字，可以是.gz或.bz2的压缩文件
# array: 存入文件的数组数据
# fmt: 写入文件的格式，例如:%d%.2f%.18e。
# delimiter: 分割字符串，默认是任何空格。
np.savetxt('a.csv', a, fmt='%d', delimiter=',')

# 其他参数使用，可参考源码中的示例
b = np.loadtxt('a.csv', dtype=np.int16, delimiter=',')

"""
多维数据的存储，读取.
需要注意：fromfile 方法读取时需要知道存入文件时数组的维度和元素类型。
即 a.tofile 和 np.fromfile 两个方法需要配合使用，可通过元数据文件来存储这些信息
"""
# frame: 文件名字符串。
# sep: 数据分割字符串，如果是空串，写入的文件为二进制格式。
# format: 写入数据的格式。
a.tofile('a.txt', sep=',', format='%d')
a.tofile('b.txt', format='%d')

# frame: 文件名字符串。
# dtype: 读取的数据类型。
# count: 读入元素的个数，-1表示读入整个文件
# sep: 数据分割字符串，如果是空串，读取文件为二进制。
b = np.fromfile('a.txt', dtype=np.int16, count=-1, sep=',')
b = b.reshape((5, 20))

"""
更方便多维数据的存储，读取的方法，很好的解决了数据维度问题（使用 numpy 自定义的数据格式）
frame: 文件名，以.npy为扩展名，压缩扩展名为.npz
array: 数组变量
np.load(fname)

"""
np.save('a.npy', a)
np.load('a.npy')
np.savez('a.npz', a)
np.load('a.npz')

# 设置随机数种子，使产生的随机数，每次都相同
np.random.seed(1)
# 生成 0-1 之间的随机小数，均匀分布，参数为 shape
np.random.rand(3, 4, 5)
# 生成指定数值之间的，指定维度的数字，类型与指定数值相同
np.random.randint(1, 10, (3, 4))


"""
# 直接修改原数组
# 当是一维数组时，会原地打乱数组的顺序，直接修改原数组
# 当是多维数组时，根据数组a的第1轴（第1行开始）进行随排列，改变数组x（行顺序改变，但行内元素顺序不变）
# shuffle(a)
"""
a = np.array([1, 2, 3, 4, 5])
# [1 2 3 4 5]

np.random.shuffle(a)
# 可能是 [3 1 5 2 4] - 原数组被修改了！

# 示例2：二维数组 - 按第1轴（行）打乱
b = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
np.random.shuffle(b)
# 可能是：
# [[4 5 6]
#  [1 2 3]
#  [7 8 9]]
# 行的顺序变了，但每行内部元素顺序不变

"""
# 创建打乱后的新数组，不改变原数组
# 当是一维数组时，返回打乱数组元素的顺序的，新数组
# 当是多维数组时，根据数组a的第1轴（第1行开始）产生一个新的乱序数组，不改变数组x（行顺序改变，但行内元素顺序不变）
# permutation(a)
"""
a = np.array([1, 2, 3, 4, 5])
# [1 2 3 4 5]

b = np.random.permutation(a)
# 可能是 [4 1 3 5 2]
# 原数组仍然是 [1 2 3 4 5]

# 示例2：直接传入整数
c = np.random.permutation(5)
# 可能是 [3 0 2 4 1] - 生成0到4的随机排列

# 示例3：二维数组
d = np.array([[1, 2], [3, 4], [5, 6]])
e = np.random.permutation(d)
# 行的顺序被打乱，但原数组d保持不变

"""
# 从给定数组中按概率随机抽样，返回新数组，不改变原数组
# 从一维数组a中以概率p抽取元素，形成 size 形状/个数 的新数组，replace表示是否可以重用元素，默认为False
# al: 数组或整数（如果是整数n，则从0到n-1中抽取）
# size：输出形状或个数，如 5、(2,3)、(2,3,4)
# replace：是否放回。True：有放回，可能抽到相同元素。False：无放回，不会抽到相同元素
# p：概率数组，长度必须与a相同，且和为1
# choice(al,size,replace,p])
"""
a = np.array([10, 20, 30, 40, 50])  # 候选数组
result1 = np.random.choice(a, size=3)
# 可能是 [20 50 10]

result2 = np.random.choice(a, size=5, replace=True)
# 可能有重复元素，如 [10 10 30 20 10]

result3 = np.random.choice(a, size=3, replace=False)
# 不会有重复元素

probabilities = [0.1, 0.1, 0.1, 0.1, 0.6]  # 即对应数字的概率，50被抽中的概率是60%
result4 = np.random.choice(a, size=10, p=probabilities)
# [50 50 50 50 50 20 50 50 50 50]

result5 = np.random.choice(a, size=(2, 3), replace=True)
# 可能是：
# [[30 20 30]
#  [50 10 40]]

"""
生成在指定范围内均匀分布的随机数。数值连续，等概率
产生具有均匀分布的数组，low起始值，high结束值，size形状/个数
"""
data = np.random.uniform(low=0, high=10, size=1000)
print("前10个数据:", data[:10])
# 可能是: [3.21, 7.89, 1.45, 9.23, 4.67, 8.91, 2.34, 6.78, 0.56, 5.43]
# 可视化
plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
plt.title("均匀分布 Uniform Distribution")
plt.xlabel("数值")
plt.ylabel("频数")
plt.show()

"""
生成正态分布（高斯分布）的随机数。连续，钟形曲线
- 钟形曲线，大部分数据集中在均值（中心峰值）附近
- 标准差（分布的离散程度）越大，数据越分散
- 自然界中很多现象都近似服从正态分布
产生具有正态分布的数组，loc均值（分布的中心位置），scale标准差（分布的离散程度），size形状/个数
"""
# 示例：标准正态分布
data_std = np.random.normal(loc=0, scale=1, size=1000)
print("标准正态分布前10个:", data_std[:10])
# 可能是: [-0.23, 1.45, -1.67, 0.89, -0.12, 2.34, -0.78, 1.23, -1.45, 0.67]

# 示例：自定义均值和标准差
data_custom = np.random.normal(loc=50, scale=10, size=1000)
print("自定义正态分布前10个:", data_custom[:10])
# 可能是: [48.7, 63.2, 42.1, 55.8, 49.3, 61.5, 45.9, 52.3, 39.8, 57.6]

# 可视化对比
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(data_std, bins=30, alpha=0.7, edgecolor='black')
plt.title("标准正态分布 (μ=0, σ=1)")
plt.xlabel("数值")

plt.subplot(1, 2, 2)
plt.hist(data_custom, bins=30, alpha=0.7, edgecolor='black')
plt.title("自定义正态分布 (μ=50, σ=10)")
plt.xlabel("数值")

plt.tight_layout()
plt.show()

"""
产生具有泊松分布的数组，用于描述单位时间内随机事件发生的次数。数值离散，非负整数
- 离散分布（只取整数值：0, 1, 2, ...）
- 描述稀有事件的发生概率
- 均值和方差都等于 lam
lam 随机事件发生率，单位时间内事件发生的平均次数
size 输出形状
poisson(lam,size)
"""
# 设置随机种子保证结果可重现
np.random.seed(42)
# 模拟每小时接到的客服电话数量（平均5通/小时）
calls_per_hour = np.random.poisson(5, 24)  # 24小时的数据
print("一天每小时电话量:", calls_per_hour)

# 模拟网站每分钟的访问量（平均20次/分钟）
visits_per_minute = np.random.poisson(20, 60)  # 60分钟的数据

# 模拟交通事故（平均每天2起）
accidents_per_day = np.random.poisson(2, 30)  # 30天的数据

# 可视化
# 创建三个子图
plt.figure(figsize=(15, 4))

# 1. 每小时电话量
plt.subplot(1, 3, 1)
plt.plot(range(24), calls_per_hour, 'o-', color='blue', markersize=4)
plt.xlabel('小时')
plt.ylabel('电话量')
plt.title('每小时电话量 (λ=5)')
plt.grid(True, alpha=0.3)

# 2. 每分钟网站访问量
plt.subplot(1, 3, 2)
plt.plot(range(60), visits_per_minute, 'o-', color='green', markersize=2)
plt.xlabel('分钟')
plt.ylabel('访问量')
plt.title('每分钟网站访问量 (λ=20)')
plt.grid(True, alpha=0.3)

# 3. 每天交通事故量
plt.subplot(1, 3, 3)
plt.bar(range(30), accidents_per_day, color='red', alpha=0.7)
plt.xlabel('天数')
plt.ylabel('事故数')
plt.title('每天交通事故量 (λ=2)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
