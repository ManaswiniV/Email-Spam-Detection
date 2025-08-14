import numpy as np
import pandas as pd

def metrics(y_true, y_pred):
    cm = np.array(pd.crosstab(y_true, y_pred, rownames = ['Actual'], colnames = ['Predicted']))
    accuracy = np.mean(y_true == y_pred)
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    return cm, accuracy, precision, recall

def plot_cm(cm):
    labels = [0, 1]
    print("    Confusion Matrix")
    print('='*25)
    print(f"{'Labels |':<10}{labels[0]:<10}{labels[1]:<10}")
    print('-'*25)
    print(f"{'':<2}{labels[0]:<5}{'|':<3}{cm[0,0]:<10}{cm[0,1]:<10}")
    print(f"{'':<2}{labels[1]:<5}{'|':<3}{cm[1,0]:<10}{cm[1,1]:<10}")

def print_report(train_cm = None, train_apr = None, test_cm = None, test_apr = None, train = True, test = True):
    if train:
        print("\nTrain Classification Report")
        print('='*55)
        print(f"Accuracy: {train_apr[0]:.4f} | Precision: {train_apr[1]:.4f} | Recall: {train_apr[2]:.4f}\n")
        plot_cm(train_cm)
    if test:
        print("\nTest Classification Report")
        print('='*55)
        print(f"Accuracy: {test_apr[0]:.4f} | Precision: {test_apr[1]:.4f} | Recall: {test_apr[2]:.4f}\n")
        plot_cm(test_cm)

def evaluate_model(model, X_train, X_test, y_train, y_test, report = False):
    model.fit(X_train, y_train)
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)
    train_cm, a, p, r = metrics(y_train, y_train_preds)
    train_apr = [a, p, r]
    test_cm, a, p, r = metrics(y_test, y_test_preds)
    test_apr = [a, p, r]
    if report:
        print_report(train_cm, train_apr, test_cm, test_apr)
    return train_apr, test_apr