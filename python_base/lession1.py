text = "LangGraph"
# Extracting a substring 正向 0，反向 1 开始
print(text[0:4])  # Outputs: Lang
print(text[-5:])  # Outputs: Graph

nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(nums[::-1])  # 基础反转        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print(nums[7::-1])  # 从索引7开始反转 [7, 6, 5, 4, 3, 2, 1, 0]
print(nums[7:2:-1])  # 从索引7到2反转 [7, 6, 5, 4, 3]
print(nums[::-2])  # 每隔一个元素反转 [9, 7, 5, 3, 1]

# 切片中负索引会被解释为从末尾开始计数，并转换为相应的正索引
print(nums[len(nums)-1:-1:-1])  # nums[9:9:-1] = []
print(nums[len(nums)-1:None:-1]) # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

nums2 = list(reversed(nums))   # 新列表 sorted()
print(nums2)
nums.reverse()   # 原地修改 .sort()
print(nums)


text = "LangGraph is a powerful framework."
# Searching for a substring
print(text.find("powerful"))  # Outputs: 15
print("Graph" in text)        # Outputs: True

# Splitting a string into a list
words = text.split(" ")
print(words)  # Outputs: ['LangGraph', 'is', 'a', 'powerful', 'framework.']

# Joining a list into a string
sentence = " ".join(words)
print(sentence)  # Outputs: LangGraph is a powerful framework.

text = "LangGraph"
print(text.upper())   # Outputs: LANGGRAPH
print(text.lower())   # Outputs: langgraph
print(text.title())   # Outputs: Langgraph


# 统计部门员工数
from collections import defaultdict
employees = [('Sales', 'Alice'), ('IT', 'Bob'), ('Sales', 'Charlie')]
dept_counts = defaultdict(int)   # 生成一个 dict, value 是 int
print(dict(dept_counts))

for dept, name in employees:   # 元组自动解包
    dept_counts[dept] += 1
print(dict(dept_counts))    # {'Sales': 2, 'IT': 1}

for i, value in enumerate(employees):   # 带索引遍历
    dept, name = value
    print(f'{i} - {dept}: {name}')


names = ['John', 'Eric']
ages = [30, 22]
for name, age in zip(names, ages):
   print(f"{name} is {age} years old.")

a = [1, 2, 3]
b = [4, 5, 6]
zipped = zip(a, b)
list(zipped) # 输出: [(1, 4), (2, 5), (3, 6)]

a1, a2 = zip(*zipped)
list(a1) # 输出: [1, 2, 3]
list(a2) # 输出: [4, 5, 6]

keys = ['name', 'age']
values = ['Alice', 24]
person_info = dict(zip(keys, values))
print(person_info) # 输出: {'name': 'Alice', 'age': 24}


TEST_NUM = 10

