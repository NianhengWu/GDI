import numpy as np
vec = np.zeros((14400, 400), dtype=np.float32)
with open('../TRAININGSET-GDI-VARDIAL2019/train.vec', 'r', encoding='utf8') as vec_file:
    for i, line in enumerate(vec_file):
        print(line)
        for j, num in enumerate(line):
            print(num)
            vec[i, j] = float(num)
