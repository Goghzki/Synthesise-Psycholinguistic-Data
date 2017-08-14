import numpy as np
from scipy import sparse, io


def get_t(words, fname):
    sparse_e = io.mmread(fname)
    e = np.array(sparse_e.todense())
    d = np.zeros((len(words), len(words)))
    for i in range(0, len(words)):
        d[i][i] = np.sum(e[i])
    sparse_d = sparse.csr_matrix(d)
    d_calculated = sparse_d.power(-0.5)
    t = (d_calculated.dot(sparse_e)).dot(d_calculated)

    return t


def get_score(fname):
    with open(fname) as f:
        words = []
        score = []
        for line in f.readlines()[1:]:
            word = line.split(",")[2]
            if word not in words:
                words.append(word)
                score.append(line.split(",")[12])
    o = np.array(score, dtype=int)
    return o


def get_seed_score(index, o):
    s = np.array(o)
    for i in range(0, len(o)):
        if i not in index:
            s[i] = 0
    return s


def get_seed_score_fb(index, o, seed):
    s = np.array(o)
    for i in range(0, len(o)):
        if i not in seed:
            if i != index:
                s[i] = 0
    return s


def propagate(s, t, beta):
    epsilon = 1e-7
    inter = 500
    p = sparse.csr_matrix(s).T
    z = sparse.csr_matrix((1 - beta) * s).T
    while inter:
        p_last = p
        p = beta * t.dot(p) + z
        if np.abs(p - p_last).sum() < epsilon:
            break
        inter = inter - 1
    return p
