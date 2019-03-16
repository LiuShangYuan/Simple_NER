import _pickle as cPickle
import numpy as np


import config
from config import train_path, train_path_i, train_path_p, test_path, test_path_i, test_path_p, valid_path, valid_path_i, valid_path_p



def parse(filepath, newpath):
    """
    加载文件并解析source和target
    :param filepath:
    :return:
    """
    word2indx = {"PAD":0, "UNK":1}
    index2word = {0:"PAD", "1":"UNK"}

    target2index = {"PAD":0}
    index2target = {0:"PAD"}

    startIndex = 2
    targetStartIndex = 1

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    sources, targets, lens = [], [], []
    current_s, current_t  = [], []
    for line in lines:
        line = line.strip()
        if line == "":
            if len(current_s) > config.maxlens: ### 过长截断
                current_s = current_s[: config.maxlens]
                current_t = current_t[: config.maxlens]
                lens.append(config.maxlens)
            else:
                lens.append(len(current_s))
                current_s.extend(["PAD"] * (config.maxlens - len(current_s)))
                current_t.extend(["PAD"] * (config.maxlens - len(current_t)))
            sources.append(current_s)
            targets.append(current_t)
            current_s, current_t  = [], []
        else:
            sw, tl = line.split("\t")
            current_s.append(sw)
            current_t.append(tl)
            if word2indx.get(sw, -1) == -1:
                word2indx[sw] = startIndex
                index2word[startIndex] = sw
                startIndex += 1
            if target2index.get(tl, -1) == -1:
                target2index[tl] = targetStartIndex
                index2target[targetStartIndex] = tl
                targetStartIndex += 1

    cPickle.dump((sources, targets, lens), open(newpath, "wb")) ##TODO 指定存储路径

    return word2indx, index2word, target2index, index2target


word2indx, index2word, target2index, index2target = parse(train_path, train_path_p)
parse(test_path, test_path_p)
parse(valid_path, valid_path_p)


cPickle.dump(word2indx, open(config.word2index_path, "wb"))
cPickle.dump(index2word, open(config.index2word_path, "wb"))
cPickle.dump(target2index, open(config.target2index_path, "wb"))
cPickle.dump(index2target, open(config.index2target_path, "wb"))


"""
处理pickle文件, 全部表示为index的形式
"""
def postprocess(picklepath, newpicklepath):
    r = cPickle.load(open(picklepath, "rb"))

    sources, targets, lens = r
    sourcesIndex, targetsIndex = [], []

    for s, t in zip(sources, targets):
        sindex, tindex = [], []
        for word, tag in zip(s, t):
            sindex.append(word2indx.get(word, 1)) ### 找到返回index, 找不到返回1(UNK)
            tindex.append(target2index[tag])
        sourcesIndex.append(sindex)
        targetsIndex.append(tindex)

    ### 保存回去
    cPickle.dump((sourcesIndex, targetsIndex, lens), open(newpicklepath, "wb"))

postprocess(train_path_p, train_path_i)
postprocess(test_path_p, test_path_i)
postprocess(valid_path_p, valid_path_i)



