# ==================== utils.py ====================
# Helpers for train/test split and evaluation

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from config import TRAIN_RATIO, RANDOM_STATE


def split_data(X, y):
    return train_test_split(X, y, train_size=TRAIN_RATIO, random_state=RANDOM_STATE)


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test).argmax(axis=1)
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))