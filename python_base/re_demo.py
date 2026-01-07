import json
import re


"""
JSON 规范严格限制控制字符（如换行 \n、回车 \r、制表符 \t 等），它们在字符串内部必须以 \n 这样的转义形式出现，不能是字面字符。
最可靠的方法是在序列化为JSON前，或解析前，批量清理和转义所有非法控制字符。
- 在写入/解析前，对字符串进行预处理 re
- 在生成数据的源头，使用 json.dumps 确保正确转义（会自动完成所有必要的转义，推荐）

SON规范允许字符串包含\n，但不建议字面换行符（即字符串的斜杠）。此模式无法跨行匹配。
需添加 re.DOTALL 标志：re.sub(pattern, func, json_str, flags=re.DOTALL)
"""
def fix_json_string(json_str):
    """
    修复JSON字符串中未转义的控制字符。
    核心原理：只匹配JSON字符串值（双引号内的部分），并对其中的控制字符进行转义。
    """
    def replace_in_quotes(match):
        # match.group(0) 是整个被匹配的字符串值，包括两端的引号
        inner_content = match.group(0)
        # 对字符串内容（去掉引号）进行转义，然后重新加上引号（含头不含尾）
        fixed = json.dumps(inner_content[1:-1])
        return fixed

    # "(?:[^"\\]|\\.)*" 匹配任意非引号、非转义字符，或一个转义序列（如\n）
    # - "" 匹配双引号内的
    # - ?: 匹配非捕获组，只匹配不提取
    # - [^] 匹配非集合内的字符
    # - \\. 匹配转义后的 \. 字符
    # 主要转义：换行(\n)、回车(\r)、制表符(\t)、退格(\b)、换页(\f)、双引号(\")、反斜杠(\\)
    # 以及其它U+0000到U-001F的控制字符（转换为\u0000形式）
    # 匹配JSON中的字符串值，每匹配到一个组，就调用指定的函数进行处理。查找并替换
    pattern = r'"(?:[^"\\]|\\.)*"'
    fixed_str = re.sub(pattern, replace_in_quotes, json_str)
    return fixed_str


# 使用示例
bad_json = '{"text": "Line1\nLine2\tTab and a \"quote\""}'
bad_json = '{"text": "第一行\n第二行", "quote": "他说：\"你好\""}'

# "Line1\nLine2\tTab and a \"quote\""
# "第一行\n第二行" "他说：\"你好\""
good_json = fix_json_string(bad_json)

data = json.loads(good_json)
print("修复并解析成功！")