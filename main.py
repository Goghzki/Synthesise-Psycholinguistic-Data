import os
import build_graph
import propagate
import evaluate
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    seed_number = input("input maximal seed number(less than 3505)")
    words = build_graph.get_words()
    if os.path.isfile('seed.txt') is False:
        seed_index = build_graph.get_random_index(seed_number, len(words))
        np.savetxt("seed.txt", seed_index, fmt='%d')
    if os.path.isfile('seed.txt') is True:
        if len(np.loadtxt("seed.txt")) != seed_number:
            seed_index = build_graph.get_random_index(seed_number, len(words))
            np.savetxt("seed.txt", seed_index, fmt='%d')
    if os.path.isfile('graph.mtx') is False:
        print "build graph"
        feature = build_graph.get_feature(words, "data.txt", 300)
        build_graph.calculate(words, feature)

    print "propagate"
    i = 100
    i_list = []
    pearson_list = []
    spearman_lsit = []
    x = []
    T = propagate.get_t(words)
    o = propagate.get_score("da.csv")

    while i <= seed_number:

        seed_index = np.loadtxt("seed.txt", dtype=int)
        '''seed_words = words[seed_index]'''

        s = propagate.get_seed_score(seed_index[0:i], o)
        P = propagate.propagate(s, T, 0.8)

        '''evaluate'''
        pearson_list.append(evaluate.get_pearson_correlation_coefficient(o, P,))
        spearman_lsit.append(evaluate.get_spearman_correlation_coefficient(o, P))
        x.append(i)
        i = i + seed_number/10

    plt.plot(x, pearson_list, 'r', label="pearson_correlation_coefficient")
    plt.plot(x, spearman_lsit, 'g', label="spearman_correlation_coefficient")
    plt.legend(loc='upper left')
    plt.xlabel('number of seed')
    plt.xlim(0, i)
    plt.show()

