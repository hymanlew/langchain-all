"""
在 Python 中，每个模块（文件）都有自己的全局命名空间。Python 没有真正的"全局变量" - 只有模块级别的变量。
在一个模块中定义的全局变量，在另一个模块中通过导入后使用，实际上是在另一个模块的命名空间中创建了一个引用。
修改这个引用会因为 可变 或 不可变类型，以及导入方式的不同而产生不同的效果。

由于Python的模块导入机制，修改的只是本文件中导入的 lesson1 模块的 TEST。对 lesson1 模块中的TEST变量实际上并没有被修改。
from lesson1 import *
TEST_NUM = 10

# 直接修改 lesson1 模块的命名空间中的变量
import lesson1
lesson1.TEST = 20
print(lesson1.TEST)
"""
TEST_NUM = 10

def __test_def():
    print("TEST DEF")

def test_abc():
    print(f'==3 {TEST_NUM}')


class MyClass:
    def __init__(self, a, b):
        self.a = a
        self.__b = b
        self.value = 1

    def show(self):
        print(f'{self.a} {self.__b}')

    # 名称修饰（Name Mangling），自动重命名为 _MyClass__var。__b 也是如此，被改名了
    # 私有化推荐使用单下划线
    def __test(self, other):
        print(f'{self.a} {self.__b}')

    def __str__(self):  # 字符串表示
        return f"MyClass({self.value})"

    def __len__(self):  # 支持 len() 函数
        return len(str(self.value))

    def __add__(self, other):  # 支持 + 运算符
        return MyClass(self.value + other.value)

    # 自定义的 __a__ 方法（不是内置的），容易与内置方法冲突，且也被当作普通函数
    # 不要这样用，私有化推荐使用单下划线
    def __a__(self):
        print("这是我的自定义特殊方法")


a = MyClass(1, 2)
# print(a.__b)
# a.__test(1)
a.__a__()
a.show()
print(a.__dict__)
print(dir(a))

