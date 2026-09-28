"""
Tune the decision thresholds of the fitted models.
"""

import pandas as pd
import lightgbm as lgb


def main():
    for nofeatures in ['', '_nofeatures']:
        for nonetwork in ['', '_nonetwork']:
            name = f'model{nofeatures}{nonetwork}'
            model = lgb.Booster(model_file=f'H:/teamSGBD/{name}.txt')
            df = pd.read_csv(f'H:/teamSGBD/eval{nofeatures}{nonetwork}.csv')

            if nofeatures == '' and nonetwork == '':
                predictions = df[['children_post2021']]

            X = df.drop(columns=['RINPERSOON', 'children_post2021'])
            probas = model.predict(X)
            predictions[name] = probas
    predictions.to_csv('H:/teamSGBD/eval_predictions.csv', index=False)


if __name__ == '__main__':
    main()
