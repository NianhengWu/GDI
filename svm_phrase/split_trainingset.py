from random import randint

indexes = [randint(0, 11400) for p in range(0, 1000)]

with open('../TRAININGSET-GDI-VARDIAL2019/train.txt', 'r', encoding='utf8')as ori:
    for i, line in enumerate(ori):
        if i in indexes:
            with open('../TRAININGSET-GDI-VARDIAL2019/split_test.txt', 'a+', encoding='utf8')as test:
                test.write(line)
        else:
            with open('../TRAININGSET-GDI-VARDIAL2019/split_train.txt', 'a+', encoding='utf8')as train:
                train.write(line)
