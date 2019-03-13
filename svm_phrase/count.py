
BS = 0
BE = 0
LU = 0
ZH = 0

with open('../train/train.txt', 'r')as traintxt:
    for each_line in traintxt:
        each_line = each_line.strip()
        label = each_line.split('\t')[1]
        if label == 'BS':
            BS += 1
        elif label == 'BE':
            BE += 1
        elif label == 'LU':
            LU += 1
        elif label == 'ZH':
            ZH += 1

print('BS', BS)  # 0.255    # 0.244
print('BE', BE)  # 0.255    # 0.244
print('LU', LU)  # 0.235    # 0.268
print('ZH', ZH)  # 0.255    # 0.244