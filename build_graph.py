import numpy as np
import random
from scipy import sparse, io


def get_words():
    with open("da.csv") as f:
        words = []
        for line in f.readlines()[1:]:
            word = line.split(",")[2]
            if word not in words:
                words.append(word)
        words = np.array(words)
    return words


def get_random_index(k, length):
    j = 0
    index = []
    while j < k:
        i = random.randint(0, length - 1)
        if i not in index:
            index.append(i)
            j = j + 1
    return index


def get_feature(words, fname, dimension):
    feature = np.zeros((len(words), dimension))
    i = len(words)
    with open(fname) as f:
        for line in f:
            if line.split()[0] in words:
                i = i - 1
                index = np.where(words == line.split()[0])[0][0]
                feature[index][:] = line.split()[1:]
            if i == 0:
                break
    return feature


def calculate(words, feature, fname):
    e = np.zeros((len(words), len(words)))
    for row in range(0, len(words)):
        for column in range(row, len(words)):
            if row < column:
                tem = (np.dot(feature[row], feature[column]))\
                      / (np.linalg.norm(feature[row])*np.linalg.norm(feature[column]))
                e[row][column] = np.arccos(-tem)
                e[column][row] = e[row][column]
    for i in range(0, len(words)):
        kth_max = np.sort(e[i])
        zero_list = [0]
        e_row = e[i]
        e_row = np.where(e_row < kth_max[len(words)-300], zero_list, e_row)
        e[i] = e_row
    for row in range(0, len(words)):
        for column in range(row, len(words)):
            if e[row][column] != e[column][row]:
                e[row][column] = 0
                e[column][row] = 0
    sparse_e = sparse.csr_matrix(e)
    io.mmwrite(fname, sparse_e)
    return
