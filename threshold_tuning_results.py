import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# load true outcome for train and holdout and predicted probabilities for train and holdout
y_train_true = pd.read_csv('') # add path
y_train_prob = pd.read_csv('') # add path

y_test_true = pd.read_csv('') # add path
y_test_prob = pd.read_csv('') # add path

# select column with true values / predicted probabilities
y_train_true = y_train_true['children_post2021'].values
y_train_prob = y_train_prob['prediction_prob'].values
y_test_true = y_test_true['children_post2021'].values
y_test_prob = y_test_true['prediction_prob'].values


res = []
for threshold in np.arange(0, 1.01, 0.01):
    row = [threshold]
    for true, prob in [[y_train_true, y_train_prob], [y_test_true, y_test_prob]]:
        for metric in [accuracy_score, precision_score, recall_score, f1_score]:
            row.append(metric(true, prob > threshold))
    res.append(row)

columns = ['threshold', 'train_accuracy', 'train_precision', 'train_recall','train_f1', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
df = pd.DataFrame(res, columns=columns)

df.to_csv('performance_across_thresholds.csv')
