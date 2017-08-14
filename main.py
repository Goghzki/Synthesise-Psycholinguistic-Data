import os
import build_graph
import propagate
import evaluate
import numpy as np
from scipy import io


if __name__ == "__main__":
    '''seed_number = input("input maximal seed number(less than 3505)")'''

    print "training..."
    train_words_number = 2000
    words = build_graph.get_words()
    if os.path.isfile('train_word_index.txt') is False:
        index = build_graph.get_random_index(train_words_number, len(words))
        np.savetxt("train_word_index.txt", index, fmt='%d')

    train_words_index = np.loadtxt("train_word_index.txt", dtype=int)
    train_words = words[train_words_index]
    if os.path.isfile('train_graph.mtx') is False:
        print "build train graph"
        feature = build_graph.get_feature(words, "data.txt", 300)
        train_feature = feature[train_words_index]
        build_graph.calculate(train_words, train_feature, "train_graph")

    seed = []
    '''seed = list(np.loadtxt("seed_index.txt", dtype=int))'''
    T = propagate.get_t(train_words, "train_graph")
    o = propagate.get_score("da.csv")
    o_train = o[train_words_index]
    beta = 0.3

    while len(seed) < 30:
        coefficient_index = -1
        i = 0
        max_coefficient = 0
        while i < train_words_number:
            if i not in seed:
                s = propagate.get_seed_score_fb(i, o_train, seed)
                P = propagate.propagate(s, T, beta)
                
                i_list = [i]
                index = seed + i_list
                coefficient = evaluate.get_pearson_correlation_coefficient(o_train, P, index)
                if coefficient > max_coefficient:
                    max_coefficient = coefficient
                    coefficient_index = i
            i = i + 1
        if coefficient_index != -1:
            seed.append(coefficient_index)
        print max_coefficient
    np.savetxt("seed_index.txt", seed, fmt='%d')

    print "testing..."

    seed_index = train_words_index[np.loadtxt("seed_index.txt", dtype=int)]
    rest_words_index = np.delete(np.array(range(0, len(words))), train_words_index)
    test_words_index = np.append(rest_words_index, seed_index)
    test_words = words[test_words_index]
    '''if os.path.isfile('test_graph.mtx') is False or len(io.mmread("test_graph.mtx").todense()) != len(test_words):'''
    print "build test graph"
    feature = build_graph.get_feature(words, "data.txt", 300)
    test_feature = feature[test_words_index]
    build_graph.calculate(test_words, test_feature, "test_graph")

    T = propagate.get_t(test_words, "test_graph")
    o_test = o[test_words_index]
    seed_number = len(seed_index)
    index = range(len(rest_words_index), (len(rest_words_index)+seed_number))
    s = propagate.get_seed_score(index, o_test)
    P = propagate.propagate(s, T, beta)

    print evaluate.get_pearson_correlation_coefficient(o_test, P, index)
