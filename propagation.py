import numpy as np
import datetime
from scipy import sparse, io, stats
import random

if __name__ == "__main__":
    print "calculating......"
    k = 100
    print k
    B = 0.8
    count = 0
    start_time = datetime.datetime.now()
    sE = io.mmread('weight_matrix')
    E = np.array(sE.todense())
    D = np.zeros((len(E[0]), len(E[0])))
    for i in range(0, len(E[0])):
        D[i][i] = np.sum(E[i])
        if D[i][i] == 0:
            count = count+1
    sD = sparse.csr_matrix(D)
    D_calculated = sD.power(-0.5)
    T = (D_calculated.dot(sE)).dot(D_calculated)
    end_time = datetime.datetime.now()
    print (end_time - start_time).seconds, "s"
    file_name = "da.csv"
    with open(file_name) as f:
        words = []
        ration = []
        for line in f.readlines()[1:]:
            word = line.split(",")[2]
            if word not in words:
                words.append(word)
                ration.append(line.split(",")[12])
    labeled_words = np.array(words)
    s = np.array(ration, dtype=int)



    o = np.array(s)
    index = []
    inter = 1000
    j = 0

    while j < k:
        i = random.randint(0, len(ration) - 1)
        if i not in index:
            index.append(i)
            j = j + 1
    for i in range(0, len(ration)):
        if i not in index:
            s[i] = 0
    '''while j < k:

        index.append(j)
        j = j + 1
    for i in range(0, len(ration)):
        if i not in index:
            s[i] = 0'''

    P = sparse.csr_matrix(s).T

    Z = sparse.csr_matrix((1 - B) * s).T

    epsilon = 1e-7
    while inter:
        P_last = P
        P = B * T.dot(P) + Z
        '''P_term = P.toarray()
        for i in range(0, len(index)):
            P_term[index[i]] = s[index[i]]'''

        P = sparse.csr_matrix(P)
        if np.abs(P - P_last).sum() < epsilon:
            break
        inter = inter - 1

    print "interation is", inter

    '''I = np.identity(len(labeled_words))
    P = (1 - B) * np.dot(np.linalg.inv((I - B * T.toarray())), s)
    '''



    "evaulation"
    '''P = P * average_rat'''

    '''print sparse.csr_matrix(P)
    print sparse.csr_matrix(o).T'''
    P = np.reshape(P.toarray(), len(labeled_words))
    total = 0



    print stats.pearsonr(P, o)
