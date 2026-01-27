from collections import defaultdict

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

doc_maps = {
    "B": 2,
    "C": 3,
    "A": 1,
}
# 不改变原数据，新列表
a = sorted(doc_maps, key=lambda x: doc_maps[x])
print(a)
print(doc_maps)

doc_maps = {
    "B": {2, 4},
    "C": {3},
    "A": {1, 5},
}
a = sorted([{k: sorted(v)} for k, v in doc_maps.items()], key=lambda d: list(d.values())[0][0])
print(a)
print(doc_maps.keys())

docs = [
    {"doc_id": "1", "id": "A"},
    {"doc_id": "3", "id": "A"},
    {"doc_id": "2", "id": "B"},
    {"doc_id": "4", "id": "A"},
    {"doc_id": "5", "id": "C"},
    {"doc_id": "1", "id": "A"}
]
# 分组去重（使用集合去重，然后转换为列表并排序）
temp_dict = defaultdict(set)
for doc in docs:
    temp_dict[doc["id"]].add(doc["doc_id"])

"""
map() 是惰性求值的：返回的是一个迭代器，不会立即执行，只有在迭代时（例如外套一个 list 收集函数）才会执行函数。
map() 通常用于转换数据，不适用于有副作用的操作：即只用于对输入的处理或计算，而不应该对外部任何变量做任何修改操作。
"""
# map(lambda doc:temp_dict[doc["id"]].add(doc["doc_id"]), docs)

# 转换为目标格式并排序
result_list = sorted(
    [{key: sorted(value, key=lambda x: int(x))}
     for key, value in temp_dict.items()],
    key=lambda d: int(list(d.values())[0][0])
)
print(result_list)
# 输出: [{'B': ['2']}, {'A': ['1', '3', '4']}, {'C': ['5']}]


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

