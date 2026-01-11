try:
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
except Exception as e:
    print(f"Error: {e}")

# Seaborn 是一个基于 matplotlib 的 Python 数据可视化库，主要用于绘制统计图形。它提供了更高级的接口，
# 能够更轻松地创建具有吸引力的统计图表，特别适合数据探索和理解数据分布、关系等。
import seaborn as sns
sns.set(style="darkgrid")
tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.show()
