from random import randint

indexes = [randint(0, 11400) for p in range(0, 1000)]
record = list()

with open('../TRAININGSET-GDI-VARDIAL2019/train.txt', 'r', encoding='utf8')as ori:
    for i, line in enumerate(ori):
        if i in indexes:
            with open('../TRAININGSET-GDI-VARDIAL2019/split_test.txt', 'a+', encoding='utf8')as test:
                test.write(line)
            record.append(i)
        else:
            with open('../TRAININGSET-GDI-VARDIAL2019/split_train.txt', 'a+', encoding='utf8')as train:
                train.write(line)


with open('../TRAININGSET-GDI-VARDIAL2019/train.vec', 'r', encoding='utf8')as ori2:
    for i, line in enumerate(ori2):
        if i in record:
            with open('../TRAININGSET-GDI-VARDIAL2019/split_test.vec', 'a+', encoding='utf8')as test:
                test.write(line)

        else:
            with open('../TRAININGSET-GDI-VARDIAL2019/split_train.vec', 'a+', encoding='utf8')as train:
                train.write(line)
