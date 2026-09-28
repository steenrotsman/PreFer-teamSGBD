import logging
from itertools import product

import numpy as np
import pandas as pd
import lightgbm as lgb

from threshold import f1

SEED = 1234
FOLDS = 10
PARAM_GRID = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'force_col_wise': True,
}

NLEAVES = [30, 50, 70, 90]
MIN_DATA = [250, 500, 1000, 2000]
NITER = [10 ** x for x in range(1, 4)]
ETA = [10 ** (-x) for x in range(1, 4)]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    for nofeatures in ['', '_nofeatures']:
        for nonetwork in ['', '_nonetwork']:
            train = pd.read_csv(f'H:/teamSGBD/train{nofeatures}{nonetwork}.csv')
            logger.info(f'Read train{nofeatures}{nonetwork}.csv')

            train_model(train, f'model{nofeatures}{nonetwork}')
            logger.info(f'Fit and saved model{nofeatures}{nonetwork}')


def train_model(df, model_name):
    data = lgb.Dataset(df.drop(columns=['RINPERSOON', 'children_post2021']), label=df.children_post2021)
    model = lgb.train(PARAM_GRID, data)
    model.save_model(f'H:/teamSGBD/{model_name}.txt')


if __name__ == '__main__':
    main()
