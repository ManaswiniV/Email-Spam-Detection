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

class KNN_Classifier:
    def __init__(self, k = 3):
        self.k = k
        self.vocabulary = None
        self.word_to_index = None

    def preprocess(self, text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\b\w{1,2}\b', '', text)
        text = re.sub(r'\d+', '', text)
        stop_words = set(STOPWORDS)
        return [word for word in text.split() if word not in stop_words]

    def tokenize(self, X):
        X_vectorized = np.zeros((len(X), len(self.vocabulary)))
        for i, msg in enumerate(X):
            for word in self.preprocess(msg):
                if word in self.vocabulary:
                    X_vectorized[i, self.word_to_index[word]] += 1
        return X_vectorized

    def distance(self, p1, p2):
        return np.sum(np.abs(p1 - p2))

    def fit(self, X, y):
        print("Training...")
        self.vocabulary = set(word for msg in X for word in self.preprocess(msg))
        print_bar(0, 4)
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocabulary)}
        print_bar(1, 4)
        self.X_train = self.tokenize(X)
        print_bar(2, 4)
        self.y_train = np.array(y).astype(int)
        print_bar(3, 4)
        print("")

    def predict(self, X):
        X_vec = self.tokenize(X)
        preds = []
        counter, total = 0, len(X)
        print("Predicting...")
        for x in X_vec:
            print_bar(counter, total)
            counter += 1
            dists = [self.distance(x, x_train) for x_train in self.X_train]
            top_k_indices = np.argsort(dists)[:self.k]
            k_nearest = self.y_train[top_k_indices]
            most_common = np.bincount(k_nearest).argmax()
            preds.append(most_common)
        print("")
        return np.array(preds)