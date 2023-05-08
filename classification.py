import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score


def load_data(filepaths):
    datasets = [pd.read_csv(filepath, index_col=0) for filepath in filepaths]
    return pd.concat(datasets)


def split_data(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def train_classifiers(X_train, y_train):
    lr_classifier = LogisticRegression(random_state=0, multi_class='multinomial')
    lr_classifier.fit(X_train, y_train)

    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)

    return lr_classifier, nb_classifier


def evaluate_classifiers(classifiers, X_test, y_test):
    results = {}
    metrics = ["accuracy", "f1_macro", "f1_micro"]

    for name, classifier in classifiers.items():
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_micro = f1_score(y_test, y_pred, average='micro')

        results[name] = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro
        }

    return results


if __name__ == "__main__":
    train_filepaths = [
        r"C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\cosine_similarities\problem00002_cosine_similarity.csv",
        r"C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\cosine_similarities\problem00004_cosine_similarity.csv",
    ]

    test_filepaths = [
        r"C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\cosine_similarities\problem00001_cosine_similarity.csv",
        r"C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\cosine_similarities\problem00003_cosine_similarity.csv",
        r"C:\Users\ltrbn\OneDrive\Desktop\CMPE493\TP\cosine_similarities\problem00005_cosine_similarity.csv",
    ]

    train_data = load_data(train_filepaths)
    test_data = load_data(test_filepaths)

    X_train, y_train = split_data(train_data)
    X_test, y_test = split_data(test_data)

    classifiers = {}
    classifiers['Logistic Regression'], classifiers['Naive Bayes (Gaussian)'] = train_classifiers(X_train, y_train)

    results = evaluate_classifiers(classifiers, X_test, y_test)

    for classifier_name, metrics in results.items():
        accuracy = metrics['accuracy']
        f1_macro = metrics['f1_macro']
        f1_micro = metrics['f1_micro']
        print(f"{classifier_name}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  F1 Macro: {f1_macro:.3f}")
        print(f"  F1 Micro: {f1_micro:.3f}")
        print()
