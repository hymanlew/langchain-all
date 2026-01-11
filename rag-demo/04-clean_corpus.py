#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
针对包含很多英文，日文，德语，中文标点，乱码等一些字符，需要把这些字符清洗掉，只留下中文字符，
仅仅留下中文字符只是一种处理方案，不同的任务需要不同的处理

以下是对数据进行简单的处理，像词向量任务，源码：
https://github.com/bamtercelboo/corpus_process_script/tree/master/clean

使用：
python clean_corpus.py –input input_file –output output_file
"""
import sys
import os
from optparse import OptionParser


class Clean(object):
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile
        self.corpus = []
        self.remove_corpus = []
        self.read(self.infile)
        self.remove(self.corpus)
        self.write(self.remove_corpus, self.outfile)

    def read(self, path):
        print("reading now......")
        if os.path.isfile(path) is False:
            print("path is not a file")
            exit()
        now_line = 0
        with open(path, encoding="UTF-8") as f:
            for line in f:
                now_line += 1
                line = line.replace("\n", "").replace("\t", "")
                self.corpus.append(line)
        print("read finished.")

    def remove(self, list):
        print("removing now......")
        for line in list:
            re_list = []
            for word in line:
                if self.is_chinese(word) is False:
                    continue
                re_list.append(word)
            self.remove_corpus.append("".join(re_list))
        print("remove finished.")

    def write(self, list, path):
        print("writing now......")
        if os.path.exists(path):
            os.remove(path)
        file = open(path, encoding="UTF-8", mode="w")
        for line in list:
            file.writelines(line + "\n")
        file.close()
        print("writing finished")

    def is_chinese(self, uchar):
        """判断一个unicode是否是汉字"""
        if (uchar >= u'\u4e00') and (uchar <= u'\u9fa5'):
            return True
        else:
            return False


if __name__ == "__main__":
    print("clean corpus")

    parser = OptionParser()
    parser.add_option("--input", dest="input", default="", help="input file")
    parser.add_option("--output", dest="output", default="", help="output file")
    (options, args) = parser.parse_args()

    input = options.input
    output = options.output

    try:
        Clean(infile=input, outfile=output)
        print("All Finished.")
    except Exception as err:
        print(err)