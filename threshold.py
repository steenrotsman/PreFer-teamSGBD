"""
Tune the decision thresholds of the fitted models.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import hmean


def main():
    for nofeatures in ['', '_nofeatures']:
        for nonetwork in ['', '_nonetwork']:
            name = f'model{nofeatures}{nonetwork}'
            model = lgb.Booster(model_file=f'H:/teamSGBD/{name}.txt')
            df = pd.read_csv(f'H:/teamSGBD/eval{nofeatures}{nonetwork}.csv')

            X = df.drop(columns=['RINPERSOON', 'children_post2021'])
            y_true = df.children_post2021
            probas = model.predict(X)
            scores = []

            for i in range(101):
                y_pred = probas > i/100
                scores.append(f1(y_pred, y_true))

            with open('H:/teamSGBD/model_results.csv', 'a') as fp:
                fp.write(f'{name},{np.argmax(scores)},{max(scores)},{str(scores)[1:-1]}\n')


def f1(y_pred, y_true):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    if precision + recall == 0:
        return 0
    return hmean([precision, recall])



if __name__ == '__main__':
    main()
