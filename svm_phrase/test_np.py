import numpy as np
from sklearn.svm import LinearSVC
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

training_set_path = '../kars/trad_train_3000.txt'
testing_set_path = '../kars/trad_test_500.txt'
sentences = []
dialects = []
features = []

sentence_features = []
n_gram_set = set()
n_value = 2

with open(training_set_path, 'r', encoding='utf8') as training_file:
    for line in training_file:
        the_sentence, the_dialect = line.strip().split('\t')
        the_sentence = the_sentence.strip()
        sentences.append('#' + the_sentence + '#')
        dialects.append(the_dialect)


def _char_n_grams(sentence):
    return [sentence[i: i + n_value] for i in range(len(sentence) - n_value + 1)]


def _word_n_grams(sentence):
    ngram = list()
    sentence = sentence.strip().split(" ")
    for i in range(len(sentence) - n_value + 1):
        gram = ""
        for j in range(n_value):
            gram = gram + sentence[i + j]
        ngram.append(gram)
    return ngram


for each in sentences:
    ngram = _char_n_grams(each)
    sentence_features.append(set(ngram))
    n_gram_set.update(ngram)


length = len(sentences)
width = len(n_gram_set)
features = np.zeros((length, width), dtype=np.int8)
for i, sent in enumerate(sentence_features):
    for j, ngram in enumerate(n_gram_set):
        if ngram in sent:
            features[i, j] = 1

model2 = LinearSVC()
model2.fit(features, dialects)
print(features, np.shape(features))


def testing():
    test_sentences = []
    test_dialects = []
    with open(testing_set_path, 'r', encoding='utf8') as test_file:
        for line in test_file:
            s, label = line.strip().split('\t')
            s = s.strip()
            test_sentences.append('#' + s + '#')
            test_dialects.append(label)

    s_feat = []
    for s in test_sentences:
        ngram = _char_n_grams(s)
        s_feat.append(set(ngram))

    test_features = np.zeros((length, width), dtype=np.int8)
    for i, s in enumerate(s_feat):
        for j, ngram in enumerate(n_gram_set):
            if ngram in s:
                test_features[i, j] = 1
    for i in range(length - len(test_sentences)):
        test_dialects.append('T')
    result = model2.predict(X=test_features)
    f1_score_T = sklearn.metrics.f1_score(test_dialects, result[:len(test_dialects)], pos_label='T', average='binary')
    f1_score_M = sklearn.metrics.f1_score(test_dialects, result[:len(test_dialects)], pos_label='M', average='binary')
    f1_score = sklearn.metrics.f1_score(test_dialects, result[:len(test_dialects)], average='macro')
    score = model2.score(test_features, test_dialects)
    accuracy = ((3000 * score - 2500) / 500) * 100
    print((f1_score_T + f1_score_M) / 2, f1_score, accuracy)

testing()