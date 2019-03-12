from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import csv


def yule_co_pairwise(result, l):
    matrix = {}

    for each in result:
        each_to_int = each[3][3][:l]
        for i, each_label in enumerate(each_to_int):
            if each_label == 'T':
                each_to_int[i] = 0
            else:
                each_to_int[i] = 1

        matrix[str(each[0])+each[2]] = each_to_int

    for key_i, each_vec in matrix.items():
        row = list([key_i])
        for key_j, other_vec in matrix.items():
            each_vec = np.array(each_vec)
            each_vec = each_vec.reshape(1, -1)
            other_vec = np.array(other_vec)
            other_vec = other_vec.reshape(1, -1)
            #sim_score = jaccard_similarity_score(each_vec, other_vec)
            sim_score = cosine_similarity(each_vec, other_vec)
            row.append(sim_score)
            print(key_i, key_j, sim_score)
        with open('jaccard_similarity_score.csv', 'a+', encoding='utf8')as f:
            writer_csv = csv.writer(f, delimiter=',')
            writer_csv.writerow(row)

