"""
切片策略（Chunking Strategy）
1，通用递归切割器（RecursiveCharacterTextSplitter）：
- 文本的拆分方式：按字符列表。
- 如何测量块大小：按字符数。

2，根据标题切割（HeaderTextSplitter）：
- 是一个结构感知的 chunker，会自动根据指定的标头进行拆分
- 要指定拆分的标头
- 适用于 word, markdown 等格式的文档

3，根据语义切割（SemanticChunker）：
- 根据 chunk 的语义相似性来拆分 chunk。如果 embeddings 相距足够远，则 chunk 将被拆分。其工作原理是通过查找任意两个句子之间的嵌入差异来完成的。
当该差异超过某个阈值时，它们将被拆分。

通过调整这些方法，可以根据特定的需求和文本特性灵活地处理和分析文本，使长文档的处理和理解变得更加高效和精确。
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter, MarkdownHeaderTextSplitter

# 1，递归切割器
with open('file/test.txt', encoding='utf8') as f:
    text_data = f.read()

'''
RecursiveCharacterTextSplitter：
chunk_size：块的最大大小，其中大小由length_function决定
chunk_overlap：数据块之间的目标重叠。重叠数据块有助于在数据块之间划分上下文时减少信息丢失。
length_function：确定块大小的函数。
is_separator_regex：指定分隔符列表是否为正则表达式，默认为 false。
separators：指定切片分隔符（从上到下顺序），默认的分隔符：["\n\n", "\n", " ", ""]
'''
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        ".",
        "?",
        "!",
        "。",
        "！",
        "？",
        ",",
        "，",
        " "
    ]
)
# 这种方式只能对 document 对象进行切割
# chunks_list = text_splitter.split_documents([text_data])
# 对于纯文本内容-字符串，则需要使用另一个函数
chunks_list = text_splitter.create_documents([text_data])
print(len(chunks_list))
print(chunks_list[0])


# 2，根据标题切割
html_string = """
<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Foo</h1>

        <p>Some intro text about Foo.</p>
        <div>
            <h2>Bar main section</h2>
            <p>Some intro text about Bar.</p>
            <h3>Bar subsection 1</h3>
            <p>Some text about the first subtopic of Bar.</p>
            <h3>Bar subsection 2</h3>
            <p>Some text about the second subtopic of Bar.</p>
        </div>
        <div>
            <h2>Baz</h2>
            <p>Some text about Baz</p>
        </div>
        <br>
        <p>Some concluding text about Foo</p>
    </div>
</body>
</html>
"""

label_split = [  # 定义章节的结构
    ('h1', '大章节 1'),
    ('h2', '小节 2'),
    ('h3', '章节中的小点 3'),
]
html_splitter = HTMLHeaderTextSplitter(label_split)
docs_list = html_splitter.split_text(html_string)
print('切割之后的结果: -------------------')
print(docs_list)

label_split_2 = [  # 定义章节的结构
    ('h1', '大章节'),
    ('h2', '小节'),
    ('h3', '章节中的小点'),
    ('h4', '小点中的子节点'),
]
html_splitter = HTMLHeaderTextSplitter(label_split_2)
docs_list = html_splitter.split_text_from_url('https://plato.stanford.edu/entries/goedel/')
print(docs_list[0])

# 如果以上按照章节切割后的内容，还是太多，即某些章节的内容过多，则可以再进行切割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",
        "\n",
        ".",
        "?",
        "!",
        "。",
        "！",
        "？",
        ",",
        "，",
        " "
    ]
)
docs2_list = text_splitter.split_documents(docs_list)
print('---------------------------------')
print(len(docs2_list))


with open('file/Foo.md', encoding='utf8') as f:
    text_data = f.read()

label_split = [
    ('#', '大章节'),
    ('##', '小节'),
    ('###', '小点')
]
# strip_headers：是否删除原 Markdown 的标题，即拆分后的文本块，是否仅包含标题下方的正文内容。
# True 是删除标题，只保留正文。False 是保留标题结构
markdown_splitter = MarkdownHeaderTextSplitter(label_split, strip_headers=False)
docs_list = markdown_splitter.split_text(text_data)
print(docs_list)


# 3，根据语义切割
"""
- 百分位数：默认的分割方法，计算所有句子间差异的百分位数（以句子-句号为单位的），任何超过 X 百分位的差异都会导致分割。
  text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")

- 标准差：基于所有句子之间，超过 X 标准差的差异进行分割。这种方法侧重于数据的波动程度，适用于需要根据数据的波动情况来决定分割点的场景。
  text_splitter= SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation")

- 四分位间距：Interquartile Range, IQR，是描述数据分布范围的一个统计量，通过计算上四分位数（前 75%）与下四分位数（后 25%）之间的差值来得到。在使用四分位距方法时，通常会将超过上四分位数加上一定倍数的IQR（或减去一定倍数的 IQR）的数据点视为异常值。
- 在文本分块的情况下，这种方法可以帮助识别和分割那些在语义嵌入空间中显著不同的句子，从而在保持文本整体连贯性的同时，区分出显著不同的部分。
  text_splitter = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="interquartile")

应用场景：
- 百分位数方法更注重数据的相对位置，适用于需要根据数据分布的整体情况，来决定分割点的场景;
- 标准差方法侧重于衡量数据的离散程度，适合于波动性较大的数据分析;
- 四分位距方法则通过识别数据的中心趋势和离散范围来确定异常值或分割点，适用于需要去除极端值或识别主体数据集的场合。

灵活性与鲁棒性：
- 标准差方法和四分位距方法在处理极端值和异常值时比百分位数方法更具鲁棒性，因为它们能够根据数据的分布特性调整分割阈值。
- 百分位数方法则严格依赖于数据的相对排名。
"""
import os
from langchain_community.embeddings import BaichuanTextEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

with open('file/test.txt', encoding='utf8') as f:
    text_data = f.read()

os.environ['BAICHUAN_API_KEY'] = 'sk-732b2b80be7bd800cb3a1dbc330722b4'
embeddings = BaichuanTextEmbeddings()

text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type='percentile')
docs_list = text_splitter.create_documents([text_data])
print(docs_list[0].page_content)

