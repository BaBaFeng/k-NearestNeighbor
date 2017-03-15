#!/usr/bin/env python
# -*- coding: utf8 -*-

# author: xiaofengfeng
# create: 2017-03-15 11:23:42

#########################################################################
# kNN: k Nearest Neighbors
# Powered by Python 3.
# Code by zouxy09: http://blog.csdn.net/zouxy09/article/details/16955347
#########################################################################

import numpy as np


def get_data():
    with open("testdata.txt", "r", encoding="utf8") as xopen:
        data = xopen.read()
        xopen.close()

    array_list = list()
    tags = list()

    line_list = data.split("\n")
    for line in line_list:
        line = line.split("\t")
        if len(line) < 4:
            continue
        array_list.append([float(l) for l in line[:3]])
        tags.append(line[-1])

    return np.array(array_list), tags


def get_euclidean_distance(arrayX, arrayY):
    return (np.sum(np.array([(a - b) ** 2 for (a, b) in zip(arrayX, arrayY)]), axis=1)) ** 0.5


def kNNClassify(test, data, tags, k):
    # step 1: calculate Euclidean distance # 第一步：计算欧式距离
    distance = get_euclidean_distance(np.tile(test, (data.shape[0], 1)), data)

    # step 2: sort the distance # 第二步：对距离排序
    sorte_dist_indices = np.argsort(distance)

    class_count_dict = dict()
    for i in range(k):
        # step 3: choose the min k distance # 第三步：选择距离最小的k个点
        vote_tag = tags[sorte_dist_indices[i]]

        # step 4: count the times labels occur # 第四步：计算在tags里出现的次数
        class_count_dict[vote_tag] = class_count_dict.get(vote_tag, 0) + 1

    # step 5: the max voted class will return # 第五步：计算出现次数最多的tag
    max_index = sorted(class_count_dict.items(), key=lambda item: item[1], reverse=True)[0]

    return max_index[0]


if __name__ == '__main__':
    data, tags = get_data()

    k = 100
    test = np.array([15360, 8.545204, 1.340429])
    tag = kNNClassify(test, data, tags, k)
    print(tag)
