import numpy as np
from sklearn.svm import LinearSVC
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
import warnings
import sys
warnings.filterwarnings('ignore')

training_set_path = '../data/simp_16770.train'
testing_set_path = '../data/simp_2000.test'
sentences = []
dialects = []
features = []
n_value = 3

with open(training_set_path, 'r', encoding='utf8') as training_file:
    for line in training_file:
        the_sentence, the_dialect = line.strip().split('\t')
        the_sentence = the_sentence.strip()
        if len(sentences) <= 3000:
            sentences.append('#' + the_sentence + '#')
            dialects.append(the_dialect)
        else:
            break

tfidf = TfidfVectorizer(ngram_range=(n_value, n_value), analyzer='char')
features = tfidf.fit_transform(sentences).toarray()
features_names = tfidf.get_feature_names()

model = LinearSVC(C=10000)
clf = CalibratedClassifierCV(model, method='sigmoid')
clf.fit(features, dialects)
# model.fit(features, dialects)

print(features, np.shape(features))
length = np.shape(features)[0]
width = np.shape(features)[1]


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


def testing():
    test_sentences = []
    test_dialects = []
    with open(testing_set_path, 'r', encoding='utf8') as test_file:
        for line in test_file:
            s, label = line.strip().split('\t')
            s = s.strip()
            if len(test_sentences) <= 500:
                test_sentences.append('#' + s + '#')
                test_dialects.append(label)
            else:
                break
    s_feat = []

    for s in test_sentences:
        ngram = _char_n_grams(s)
        s_feat.append(set(ngram))

    test_features = np.zeros((length, width), dtype=np.int8)

    for i, s in enumerate(s_feat):
        for j, ngram in enumerate(features_names):
            if ngram in s:
                test_features[i, j] = 1

    result = clf.predict(X=test_features)
    f1_score = sklearn.metrics.f1_score(test_dialects, result[:len(test_dialects)], average='macro')
    y_proba = clf.predict_proba(test_features)
    print(f1_score, y_proba, clf.classes_)

testing()
