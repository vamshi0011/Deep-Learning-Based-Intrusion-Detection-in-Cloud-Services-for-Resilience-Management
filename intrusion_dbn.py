import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dataset splits
splits = [('train_80.csv', 'test_20.csv'), ('train_70.csv', 'test_30.csv')]

# Preprocess function
def preprocess_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train.columns = [col.strip("'") for col in train.columns]
    test.columns = [col.strip("'") for col in test.columns]

    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

    categorical_cols = ['protocol_type', 'service', 'flag']
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Train DBN
def train_dbn(X_train, y_train, X_test, y_test):
    rbm = BernoulliRBM(n_components=32, learning_rate=0.01, n_iter=20, verbose=True)
    logistic = LogisticRegression(max_iter=1000)
    dbn = Pipeline([('rbm', rbm), ('logistic', logistic)])

    dbn.fit(X_train, y_train)
    y_pred = dbn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"DBN Accuracy: {accuracy}")
    return accuracy

# Execute
results = []
for train_file, test_file in splits:
    X_train, X_test, y_train, y_test = preprocess_data(train_file, test_file)
    accuracy = train_dbn(X_train, y_train, X_test, y_test)
    results.append({'split': train_file, 'accuracy': accuracy})

pd.DataFrame(results).to_csv('dbn_results_comparison.csv', index=False)
