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


def kNNClassify(test, data, tags, k):
    num_of_row = data.shape[0]  # 0轴的长度(记录数) # shape[0] stands for the num of row

    # step 1: calculate Euclidean distance
    # tile(A, reps): Construct an array by repeating A reps times
    # the following copy numSamples rows for dataSet
    # 第一步：计算欧式距离
    diff = np.tile(test, (num_of_row, 1)) - data  # 计算差值 # Subtract element-wise
    square_diff = diff ** 2  # 差值的平方 # squared for the subtract
    square_dist = np.sum(square_diff, axis=1)  # 记录元素之和 # sum is performed by row
    distance = square_dist ** 0.5  # 取二次根式

    # step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    # 第二步：对距离排序
    sorte_dist_indices = np.argsort(distance)

    class_count_dict = {}  # define a dictionary (can be append element)
    for i in range(k):
        # step 3: choose the min k distance
        # 第三步：选择距离最小的k个点
        vote_tag = tags[sorte_dist_indices[i]]

        # step 4: count the times labels occur
        # when the key voteLabel is not in dictionary class_count_dict, get()
        # will return 0
        # 第四步：计算在tags里出现的次数
        class_count_dict[vote_tag] = class_count_dict.get(vote_tag, 0) + 1

    # step 5: the max voted class will return
    # 第五步：计算出现次数最多的tag
    max_count = 0
    for key, value in class_count_dict.items():
        if value > max_count:
            max_count = value
            max_index = key

    return max_index


if __name__ == '__main__':
    data = np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    tags = ["A", "A", "B", "B"]

    k = 3

    test = np.array([1.0, 1.1])
    tag = kNNClassify(test, data, tags, k)
    print("Input:", test, "classified:", tag)

    test = np.array([0.0, 0.2])
    tag = kNNClassify(test, data, tags, k)
    print("Input:", test, "classified:", tag)

    test = np.array([0.8, 1.2])
    tag = kNNClassify(test, data, tags, k)
    print("Input:", test, "classified:", tag)
