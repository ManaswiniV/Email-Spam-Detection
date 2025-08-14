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

class Support_Vector_Classifier:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.weights = None
        self.bias = 0
        self.lr = learning_rate
        self.epochs = epochs
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

    def fit(self, X, y):
        self.vocabulary = set(word for msg in X for word in self.preprocess(msg))
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocabulary)}
        X_vec = self.tokenize(X)
        num_samples, num_features = X_vec.shape
        self.weights = np.zeros(num_features)
        counter =  0
        print("Training...")
        for _ in range(self.epochs):
            print_bar(counter, self.epochs)
            counter += 1
            for i, x in enumerate(X_vec):
                linear_output = x @ self.weights + self.bias
                y_pred = 1 if linear_output >= 0 else 0
                update = self.lr * (y[i] - y_pred)
                self.weights += update * x
                self.bias += update
        print("")

    def predict(self, X):
        print("Predicting...")
        X_vec = self.tokenize(X)
        print_bar(0, 2)
        linear_output = X_vec @ self.weights + self.bias
        print_bar(1, 2)
        print("")
        return np.where(linear_output >= 0, 1, 0)