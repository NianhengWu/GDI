import numpy as np
from sklearn.svm import LinearSVC
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
#from svm_phrase import fusion_methods
import fusion_methods
import warnings
from sklearn.svm import SVC
warnings.filterwarnings('ignore')


class Classifiers:
    """
    Usage:
           1. Generating models:
           After creating instances of this class, invoke the method 'training'
           with the path as its parameter, and after that, when you want to get
           the resulting model, just invoke its class variable 'model'

           2. Test model:
           Invoking 'testing' function with path to the test set as parameter,
           it will return f1 score
    """
    def __init__(self, n_value, intercept, mode='character'):
        self.n_value = n_value
        self.mode = mode
        if self.mode != 'character' and self.mode != 'word' and self.mode != 'audio':
            raise ValueError('the mode has to be either character or word')
        self.features = None
        self.dialects = list()
        self.features_names = list()
        self.length = 0
        self.width = 0
        self.model = None
        self.clf = None
        self.intercept = intercept
        self.test_dialects = None

    def _char_n_grams(self, sentence):
        return [sentence[i: i + self.n_value] for i in range(len(sentence) - self.n_value + 1)]

    def _word_n_grams(self, sentence):
        ngram = list()
        sentence = sentence.strip().split(" ")
        for i in range(len(sentence) - self.n_value + 1):
            gram = ""
            for j in range(self.n_value):
                gram = gram + sentence[i + j]
            ngram.append(gram)
        return ngram

    def training(self, training_set_path):

        sentences = list()

        with open(training_set_path, 'r', encoding='utf8') as training_file:
            for line in training_file:
                the_sentence, the_dialect = line.strip().split('\t')
                the_sentence = the_sentence.strip()
                sentences.append('#' + the_sentence + '#')
                self.dialects.append(the_dialect)

        if self.mode == 'character':
            tfidf = TfidfVectorizer(ngram_range=(self.n_value, self.n_value), analyzer='char')
            self.features = tfidf.fit_transform(sentences).toarray()
            self.features_names = tfidf.get_feature_names()

        elif self.mode == 'word':
            tfidf = TfidfVectorizer(ngram_range=(self.n_value, self.n_value), analyzer='word')
            self.features = tfidf.fit_transform(sentences).toarray()
            self.features_names = tfidf.get_feature_names()

        elif self.mode == 'audio':
            self.features = np.zeros((len(sentences), 400), dtype=np.float32)
            with open('../train/train (copy).vec', 'r', encoding='utf8') as vec_file:
                for i, line in enumerate(vec_file):
                    #print(line)
                    line = line.split(' ')
                    for j, num in enumerate(line):
                        #print(num)
                        self.features[i, j] = float(num)

        self.length = len(sentences)
        self.width = self.features.shape[1]
        self._svc()

    def _svc(self):
        # self.model = SVC(kernel='linear', C=1000)
        self.model = LinearSVC(C=2, class_weight={'BS':0.244, 'BE':0.244, 'LU':0.268, 'ZH':0.244})
        self.clf = CalibratedClassifierCV(self.model, method='sigmoid')
        self.clf.fit(self.features, self.dialects)
        self.model.fit(self.features, self.dialects)

    def testing(self, testing_set_path):
        test_sentences = []
        self.test_dialects = []
        with open(testing_set_path, 'r', encoding='utf8') as test_file:
            for line in test_file:
                s, label = line.strip().split('\t')
                s = line.strip()
                test_sentences.append('#' + s + '#')
                self.test_dialects.append(label)

        s_feat = []

        if self.mode == 'audio':
            test_features = np.zeros((self.length, 400), dtype=np.float32)
            with open('../train/dev.vec', 'r', encoding='utf8') as vec_file:
                for i, line in enumerate(vec_file):
                    line = line.split(' ')
                    for j, num in enumerate(line):
                        test_features[i, j] = float(num)
        else:
            if self.mode == 'character':
                for s in test_sentences:
                    ngram = self._char_n_grams(s)
                    s_feat.append(set(ngram))
            elif self.mode == 'word':
                for s in test_sentences:
                    ngram = self._word_n_grams(s)
                    s_feat.append(set(ngram))

            test_features = np.zeros((self.length, self.width), dtype=np.float32)

            for i, s in enumerate(s_feat):
                for j, ngram in enumerate(self.features_names):
                    if ngram in s:
                        test_features[i, j] += 1

        result = self.model.predict(X=test_features)
        #f1_score = 0
        f1_score = sklearn.metrics.f1_score(self.test_dialects, result[:len(self.test_dialects)], average='macro')

        probability_matrix, label = fusion_methods.mean_probability_rule(test_features, self.clf)

        # score = self.model.score(test_features, test_dialects)
        # accuracy = ((3000 * score - 2500) / 2000) * 100

        return f1_score, probability_matrix, label, result
        #return f1_score, label, resulttest

    def get_test_dialects(self):
        return self.test_dialects
