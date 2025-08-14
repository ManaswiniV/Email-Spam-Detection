import numpy as np
import string
import re
import sys
from collections import Counter, defaultdict
from wordcloud import STOPWORDS

# Utility Function that shows the progress
def print_bar(count, total):
    percent = int(((count + 1) / total) * 100)
    bar = '#' * (percent // 2) + ' ' * (50 - percent // 2)
    sys.stdout.write(f'\r[{bar}] {percent}%')
    sys.stdout.flush()

class Multinomial_NB_Classifier:
    def __init__(self, alpha = 1):
        self.alpha = alpha
        self.vocabulary = set()
        self.classes = None
        self.class_word_counts = defaultdict(lambda: defaultdict(int))
        self.class_counts = defaultdict(int)
        self.class_priors = defaultdict(float)

    def preprocess(self, text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\b\w{1,2}\b', '', text)
        text = re.sub(r'\d+', '', text)
        stop_words = set(STOPWORDS)
        return [word for word in text.split() if word not in stop_words]

    def fit(self, X, y):
        self.classes, self.class_counts = np.unique(y, return_counts = True)
        counter, total = 0, len(y)
        print("Training...")
        for msg, label in zip(X, y):
            print_bar(counter, total)
            counter += 1
            words = self.preprocess(msg)
            for word in words:
                self.class_word_counts[label][word] += 1
                self.vocabulary.add(word)
        print("")
        self.class_priors = {c: count / total for c, count in zip(self.classes, self.class_counts)}

    def predict(self, X):
        preds = []
        counter, total = 0, len(X)
        print("Predicting...")
        vocab_size = len(self.vocabulary)
        for msg in X:
            print_bar(counter, total)
            counter += 1
            words = self.preprocess(msg)
            class_scores = {}
            for c in self.classes:
                prob = self.class_priors[c]
                total_words = sum(self.class_word_counts[c].values())
                for word in words:
                    word_count = self.class_word_counts[c].get(word, 0)
                    word_prob = (word_count + self.alpha + 1e-10) / (total_words + vocab_size * self.alpha)
                    prob *= word_prob
                class_scores[c] = prob
            preds.append(max(class_scores, key = class_scores.get))
        print("")
        return np.array(preds)