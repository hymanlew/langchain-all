#Python Programming Concepts useful in LangGraph

#9. Unpacking a tuple
coordinates = (10, 20)
x, y = coordinates
print(x)  # Output: 10
print(y)  # Output: 20

#10. Unpacking a list
names = ["Alice", "Bob", "Charlie"]
first, second, third = names
print(first)  # Output: Alice

#11. Excess unpacking 
numbers = [1, 2, 3, 4, 5]
first, *middle, last = numbers
print(middle)  # Output: [2, 3, 4]

#23. Lambda functions
add = lambda x, y: x + y
print(add(3, 5))  # Output: 8

# map 作用是将一个函数应用到一个可迭代对象（如列表、元组等）的每个元素上，并返回一个迭代器
# 作用是对可迭代对象中的每个元素应用指定的函数，并返回一个新的迭代器。
numbers = [1, 2, 3, 4]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # Output: [1, 4, 9, 16]

words = ["hello", "world", "langgraph"]
# Using map to convert each string to uppercase
uppercase_words = list(map(lambda word: word.upper(), words))
print(uppercase_words)  # Output: ['HELLO', 'WORLD', 'LANGGRAPH']

users = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30},
    {"name": "Charlie", "age": 35}
]
# Using map to extract only the 'name' field from each dictionary
names = list(map(lambda user: user["name"], users))
print(names)  # Output: ['Alice', 'Bob', 'Charlie']

names = ["Alice", "Bob", "Charlie", "Dave"]
# Using filter to keep names with more than 3 characters
long_names = list(filter(lambda name: len(name) > 3, names))
print(long_names)  # Output: ['Alice', 'Charlie', 'Dave']

# List of dictionaries with user data
users = [
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 20},
    {"name": "Charlie", "age": 35}
]
# Using filter to keep only users who are 30 years or older
adult_users = list(filter(lambda user: user["age"] >= 30, users))
print(adult_users)  # Output: [{'name': 'Charlie', 'age': 35}]


import random
# Generate a random integer between 1 and 100
random_number = random.randint(1, 100)
print(random_number)
# Choose a random item from a list
options = ["apple", "banana", "cherry"]
fruit = random.choice(options)
print(fruit)
# 随机打乱列表中的元素顺序。需要注意，shuffle()函数会直接修改原列表，而不是返回一个新的打乱顺序的列表，
# 它的返回值是None。
numbers = [1, 2, 3, 4, 5]
shuffle = random.shuffle(numbers)
print(shuffle)
# 生成一个0到1之间的随机浮点数。uniform()函数会返回指定范围内的一个随机浮点数，范围包含下限但不包含上限（即[0.0, 1.0)）。
random_float = random.uniform(0, 1)
print(random_float)

#76. DateTime
from datetime import datetime, timedelta
# Get the current date and time
now = datetime.now()
# Calculate a future date by adding a timedelta
future_date = now + timedelta(days=5)
print(future_date.strftime("%Y-%m-%d"))

#77. Collections
from collections import Counter, defaultdict
# Counting occurrences of items in a list
words = ["apple", "banana", "apple", "cherry"]
word_count = Counter(words)
print(word_count)
# Output: Counter({'apple': 2, 'banana': 1, 'cherry': 1})

# Using defaultdict to handle missing keys
fruits = defaultdict(int)
print(fruits) # Output: defaultdict(<class 'int'>, {})
fruits["apple"] += 1
print(fruits)  # Output: defaultdict(<class 'int'>, {'apple': 1})

#78. itertools 是Python标准库中的一个模块，提供了许多用于创建迭代器的函数
from itertools import cycle
# Cycling through a list indefinitely
# 创建一个无限循环迭代器。cycle函数接收一个可迭代对象（列表等集合），并返回一个迭代器，该迭代器会无限次地重复这些元素
colors = cycle(["red", "green", "blue"])
for _ in range(6):
    print(next(colors)) # 每次调用时从迭代器中获取下一个元素

#81. Reduce 累积处理函数（加、减、乘、除）
from functools import reduce
# List of numbers to sum
numbers = [1, 2, 3, 4]
# Using reduce to sum all numbers
total_sum = reduce(lambda x, y: x + y, numbers)
print(total_sum)  # Output: 10

product = reduce(lambda x, y: x * y, numbers)
print(product)  # Output: 24

# List of strings
words = ["hello", "world", "langgraph", "is", "awesome"]
# Using reduce to find the longest string in the list
longest_word = reduce(lambda x, y: x if len(x) > len(y) else y, words)
print(longest_word)  # Output: 'langgraph'

# Filtering even numbers, mapping to squares, and reducing to sum
sum_of_squares_of_evens = reduce(
    lambda x, y: x + y,
    map(lambda x: x**2, filter(lambda x: x % 2 == 0, numbers))
)
print(sum_of_squares_of_evens)  # Output: 56 (2^2 + 4^2 + 6^2)

# 使用operator模块中的函数替代lambda表达式
import operator
sum_with_operator = reduce(operator.add, numbers)
product_with_operator = reduce(operator.mul, numbers)
print(f"使用operator.add: {sum_with_operator}")  # 输出: 15
print(f"使用operator.mul: {product_with_operator}")  # 输出: 120


#24. Function decorators
def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log_decorator
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# Output:
# Executing greet
# Hello, Alice!

#25. Recursive function
def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(5))  # Output: 120

#26.Documenting your code
def greet(name):
    """Greets a person by name."""
    print(f"Hello, {name}!")

help(greet) # Output:
# Help on function greet in module __main__:
# greet(name)
#    Greets a person by name.

#28. Creating dictionaries
#Using curly braces
user_info = { "name": "Alice", "age": 30 }
#Using the dict() constructor
user_info = dict([("name", "Alice"), ("age", 30)])
user_info = dict(name="Alice", age=30)
#Using dict.fromkeys
keys = ["name", "age", "is_active"]
user_info = dict.fromkeys(keys, None)
print(user_info)
# Output: {'name': None, 'age': None, 'is_active': None}

#30. Alternative way to acess a dictionary with default value
print(user_info.get("location", "Unknown"))  # Output: Unknown

#31. Adding a dictionary
user_info["location"] = "Wonderland"  # Adds a new key-value pair
user_info["age"] = 31                # Updates the value for the existing key
print(user_info)
# Output: {'name': 'Alice', 'age': 31, 'location': 'Wonderland'})

#32: Removing items from the dictionary
age = user_info.pop("age")
print(age)  # Output: 31
print(user_info)
# Output: {'name': 'Alice', 'location': 'Wonderland'}

del user_info["location"]
del user_info  # Deletes the entire dictionary

#last_item = user_info.popitem()
#user_info.clear()  # Removes all items from the dictionary

#33. Dictionary methods
user_info = {"name": "James", "age": 45}
keys = user_info.keys()      # Returns a view of the keys
print(keys)
values = user_info.values()  # Returns a view of the values
print(values)
items = user_info.items()    # Returns a view of the key-value pairs
print(items)

#35. Updating a dictionary
user_info.update({"location": "Nairobi", "age": 32})
print(user_info)

#Retrieve or set default value
user_info.setdefault("is_active", True)

#39. Comprehension
# Creating a dictionary with squares of numbers
squares = {x: x ** 2 for x in range(5)}
print(squares)
# Output: {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

#40. Copying dictionaries
user_info = {"name": "James", "location": {"a": "a", "b": "b"}}
user_info_copy = user_info.copy()
user_info_copy["location"].pop("a")
print(user_info["location"])  # Output: {'b': 'b'}
print(user_info_copy["location"])  # Output: {'b': 'b'}

#Nested copy
import copy
deep_copy = copy.deepcopy(user_info)
print(user_info) # Output: {'name': 'James', 'location': {'a': 'a', 'b': 'b'}}
print(deep_copy) # Output: {'name': 'James', 'location': {'b': 'b'}}

#41. Merging two dictionaries
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}
# Merging
merged = dict1 | dict2
# Output: {'a': 1, 'b': 3, 'c': 4}
# In-place merge
dict1 |= dict2
# Output: {'a': 1, 'b': 3, 'c': 4}


#42. TypedDict
from typing_extensions import TypedDict
class Person(TypedDict):
    name: str
    age: int

person : Person = {"name": "Alice", "age": 30}
print(person["name"])  # Output: Alice

#Factory method
Person = TypedDict('Person', {'id': int, 'name': str, 'age': int})

#43. Nested Dictionaries with TypedDict
class Address(TypedDict):
    street: str
    city: str
    zip_code: int

class UserProfile(TypedDict):
    username: str
    email: str
    address: Address

profile: UserProfile = {
    "username": "johndoe",
    "email": "johndoe@example.com",
    "address": {
        "street": "123 Elm St",
        "city": "Metropolis",
        "zip_code": 12345
    }
}

#54. Multiple inheritance
class Walker:
    def walk(self):
        return "Walking..."

class Swimmer:
    def swim(self):
        return "Swimming..."

class Amphibian(Walker, Swimmer):
    pass

frog = Amphibian()
print(frog.walk())  # Output: Walking...
print(frog.swim())  # Output: Swimming...

#55. Encapsulation and access modifiers
class BankAccount:
    def __init__(self, owner, balance):
        self.owner = owner
        self._balance = balance  # Protected
        self.__private_balance = balance  # private

    def __withdraw(self, amount):  # Private method
        if amount <= self._balance:
            self._balance -= amount
            return self._balance
        return "Insufficient funds"

account = BankAccount("Alice", 1000)
print(account._balance)  # Accessing a protected attribute (not recommended)
# print(account.__private_balance)  # can not call, it's private
# account.__withdraw(500) # can not call, it's private

#57. Magic methods
class Car:
    def __init__(self, make, model, x=1):
        self.make = make
        self.model = model
        self.x = x

    def __str__(self):
        return f"{self.make} {self.model}"

    def __repr__(self):
        return f"Car('{self.make}', '{self.model}')"

    def __add__(self, other):
        return Car(self.make, self.model, self.x + other.x)

    @property
    def area(self):
        return self.x * self.x

car = Car("Toyota", "Corolla")
print(car)  # Output: Toyota Corolla, toString
print(repr(car))  # Output: Car('Toyota', 'Corolla'), toString

car2 = Car("Toyota", "Corolla", 2)
result = car + car2
print(result.x)  # Output: 3
print(result.area)  # Output: 9

#60 Class properties controlling attributes
class Rectangle:
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value > 0:
            self._width = value
        else:
            raise ValueError("Width must be positive")

re = Rectangle(5, 10)
re.width(20)
print(re.width)

#61. Type hints
from typing import List, Dict, Optional
def process_data(data: Dict[str, Optional[List[int]]]) -> None:
    print(data)

process_data({"data": [1, 2, 3]})
process_data({"data": None})
process_data({"data": []})
process_data({})
print(process_data) # Output: <function process_data at 0x...>


#63 HTTP Requests
import requests
try:
    response = requests.get("https://api.example.com/data")
    if response.status_code == 200:
        data = response.json()
        print(data)
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")


#64. Pydantic
from pydantic import BaseModel, Field, ValidationError
class User(BaseModel):
    name: str
    age: int = Field(..., gt=0)

try:
    user = User(name="Alice", age=-1)
except ValidationError as e:
    print(e)

#65. Logging
import logging
# Configure basic logging settings
logging.basicConfig(level=logging.INFO)  # Sets logging to capture INFO and above messages
# Create a logger instance
logger = logging.getLogger(__name__)
# Log an informational message
logger.info("This is an informational message.")

#66. SubProcess
import sys
import subprocess
result = subprocess.run(["echo", "Hello, LangGraph!"], capture_output=True, text=True, executable=sys.executable)
print(result.stdout)


#68. Pandas
import pandas as pd
try:
    df = pd.read_csv("data.csv")
    print(df.head()) # 是取出并显示 df 的前5行数据
except FileNotFoundError:
    print("File not found")

#68. Matplotlib
try:
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()
except Exception as e:
    print(f"Error: {e}")

#69. Seaborn 是一个基于 matplotlib 的 Python 数据可视化库，主要用于绘制统计图形。它提供了更高级的接口，
# 能够更轻松地创建具有吸引力的统计图表，特别适合数据探索和理解数据分布、关系等。
import seaborn as sns
sns.set(style="darkgrid")
tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.show()


#70. sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
try: 
    engine = create_engine("sqlite:///data.db")
    Session = sessionmaker(bind=engine)
    session = Session()
except Exception as e:
    print(f"Error: {e}")

#67. json
import json
data = {"name": "Alice", "age": 30}
with open("data.json", "w") as f:
    json.dump(data, f)

# Reading from a JSON file
with open("data.json", "r") as f:
    loaded_data = json.load(f)
print(loaded_data)  # Output: {'name': 'Alice', 'age': 30}

#72. yaml file handling，pip install PyYAML
import yaml
# Writing to a YAML file
config = {"name": "Alice", "roles": ["admin", "user"]}
try:
    with open("config.yaml", "w") as f:
        yaml.dump(config, f)

    # Reading from a YAML file
    with open("config.yaml", "r") as f:
        loaded_config = yaml.safe_load(f)
    print(loaded_config)  # Output: {'name': 'Alice', 'roles': ['admin', 'user']}
except Exception as e:
    print(f"Error: {e}")

#73. Operating system file handling
import os
import sys
try:
    #Create a folder
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Get an environment variable
    api_key = os.environ.get("API_KEY", "default_value")
except Exception as e:
    print(f"Error: {e}")

# Check if enough arguments were passed
if len(sys.argv) < 2:
    print("Usage: python script.py <filename>")
    #sys.exit("Please provide a file path as an argument.")
else:
    # Accessing command-line arguments
    filename = sys.argv[1]
    print(f"Processing file: {filename}")

# 将一个自定义模块目录添加到Python的模块搜索路径中
# os.getcwd()获取当前工作目录，os.path.join()将当前工作目录与"custom_modules"组合成一个完整的路径。
custom_module_path = os.path.join(os.getcwd(), "custom_modules")

# 检查刚刚创建的路径是否已经在Python的模块搜索路径sys.path中。如果不在，就使用sys.path.append()将其添加到搜索路径中
# 目的是让Python能够在custom_modules目录中查找和导入模块。当你使用import语句时，Python会在sys.path列出的所有目录中查找模块文件
if custom_module_path not in sys.path:
    sys.path.append(custom_module_path)

# Now you can import custom modules from the added path
try:
    import custom_modules
    print("Custom module imported successfully.")
except ImportError:
    print("Failed to import custom module.")

# Get the absolute path of the current file
os.path.abspath(__file__)
# Get the parent directory of the current file
os.path.dirname(os.path.abspath(__file__))
# Get the grandparent directory of the current file
os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"
# Get the current working directory
os.getcwd()

"""
os.getcwd()
 - 返回Python进程启动时所在的目录（当前工作目录）
 - 这个路径可能与脚本文件所在的目录不同
 - 如果你从其他目录运行这个脚本，os.getcwd()会返回你运行命令的目录，而不是脚本所在的目录

os.path.dirname(os.path.abspath(__file__))
 - 返回当前Python脚本文件所在的目录
 - __file__是当前脚本的文件路径
 - os.path.abspath(__file__)获取脚本的绝对路径
 - os.path.dirname()获取该文件路径的目录部分

假设你的项目结构如下：
/home/user/myproject/
├── main.py
└── subfolder/
    └── lesson2.py

如果你在/home/user/目录下运行命令：
python myproject/subfolder/lesson2.py

- os.getcwd()会返回/home/user/（你运行命令的目录）
- os.path.dirname(os.path.abspath(__file__)) 会返回 /home/user/myproject/subfolder/（脚本实际所在的目录）
即在处理文件路径时，使用第二种更加可靠，因为它总是返回脚本文件所在的目录，而不会受到运行脚本的位置影响。
"""


#83. Aync basics
import asyncio
#Sequential tasks
async def task1():
    await asyncio.sleep(1)
    print("Task 1 completed")

async def task2():
    await asyncio.sleep(2)
    print("Task 2 completed")

async def main():
    await task1()
    await task2()
    results = await asyncio.gather(task1(), task2())
    print(results)

asyncio.run(main())
print(result)  # Output: Data fetched!

#Concurrent tasks
async def task(name):
    # 添加随机延迟，模拟实际任务的不同执行时间
    delay = random.uniform(0.5, 2.0)
    await asyncio.sleep(delay)
    print(f"Task {name} completed after {delay:.2f} seconds")

async def main():
    await asyncio.gather(task("A"), task("B"), task("C"))

asyncio.run(main())
#Output: 完成顺序并不保证。asyncio.gather()会并发运行这些任务，但由于操作系统的调度和其它因素，实际完成顺序是不确定的
# Task A completed
# Task B completed
# Task C completed

#Concurrent IO
async def fetch_source(source):
    print(f"Fetching from {source}...")
    await asyncio.sleep(2)
    print(f"Completed fetching from {source}")

async def main():
    sources = ["Source 1", "Source 2", "Source 3"]
    # * 是解包操作符，它将生成器产生的所有任务作为独立的参数传递给 asyncio.gather() 函数
    # 因为 for 循环会生成多个任务，* 操作符将这些任务作为参数传递给 asyncio.gather() 函数
    await asyncio.gather(*(fetch_source(source) for source in sources))

    tasks = [fetch_source(source) for source in sources]
    await asyncio.gather(*tasks)

asyncio.run(main())
#Output
# Fetching from Source 1...
# Completed fetching from Source 1
# Fetching from Source 2...
# Completed fetching from Source 2
# Fetching from Source 3...
# Completed fetching from Source 3

#error hanndling async
async def faulty_task():
    try:
        raise ValueError("An error occurred")
    except ValueError as e:
        print(f"Error: {e}")

asyncio.run(faulty_task())
#Output Error: An error occurred

async def task1():
    raise ValueError("Error in Task 1")

async def task2():
    await asyncio.sleep(1)
    return "Task 2 completed successfully"

async def main():
    results = await asyncio.gather(task1(), task2(), return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            print(f"Handled exception: {result}")
        else:
            print(result)

asyncio.run(main())
#Output
# Error: ValueError("Error in Task 1")
# Task 2 completed successfully
