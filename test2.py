from collections import defaultdict

# 统计部门员工数
employees = [('Sales', 'Alice'), ('IT', 'Bob'), ('Sales', 'Charlie')]
dept_counts = defaultdict(int)
print(dept_counts)
print()

for dept, name in employees:
    dept_counts[dept] += 1
print(dict(dept_counts))
# {'Sales': 2, 'IT': 1}
