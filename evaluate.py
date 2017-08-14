from scipy import stats, sparse
import numpy as np


def get_pearson_correlation_coefficient(o, p, index):
    if type(p) == sparse.csr_matrix:
        p = np.reshape(p.toarray(), len(o))
    o = np.array(o, dtype=float)
    p = np.delete(p, index)
    o = np.delete(o, index)
    return stats.pearsonr(o, p)[0]


def get_spearman_correlation_coefficient(o, p):
    if type(p) == sparse.csr_matrix:
        p = np.reshape(p.toarray(), len(o))
    return stats.spearmanr(o, p)[0]
