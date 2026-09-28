import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from add_features import IDS, add_features, clean
from network import add_network
from threshold import f1
from train import train_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NETWORKS = ["_BUREN", "_FAMILIE", "COLLEGA"]
BASE = ["RINPERSOON", "children_post2021"]

path_to_holdout_data = ""
group_name = "teamSGBD"


def main():
    train = pd.read_csv("H:/teamSGBD/train.csv", dtype={"RINPERSOON": str})
    eval = pd.read_csv("H:/teamSGBD/eval.csv", dtype={"RINPERSOON": str})
    full = pd.read_csv("H:/teamSGBD/full.csv", dtype={"RINPERSOON": str})

    fit_additional(train, eval)
    predict_outcomes(path_to_holdout_data, group_name, full, train)


def fit_additional(train, eval):
    y_true = eval.children_post2021
    for name, cols in get_additional(train).items():
        # Train model
        train_model(train[cols], name)
        model = lgb.Booster(model_file=f"H:/teamSGBD/{name}.txt")
        X = eval[cols].drop(columns=BASE)
        probas = model.predict(X)

        # Tune decision threshold
        scores = []
        for i in range(101):
            y_pred = probas > i / 100
            scores.append(f1(y_pred, y_true))
        best_threshold = np.argmax(scores)
        with open("H:/teamSGBD/model_results.csv", "a") as fp:
            fp.write(f"{name},{best_threshold},{max(scores)},{str(scores)[1:-1]}\n")


def predict_outcomes(path_to_holdout_data, group_name, full, train):
    # Load the data
    df = pd.read_csv(
        path_to_holdout_data, low_memory=False, dtype={col: str for col in IDS}
    )

    # Save IDs to a new df
    predictions = df[["RINPERSOON"]]

    # Add our engineered features and network features, takes quite long
    df = add_network(clean(add_features(df), train), full)

    for name, cols in get_additional(train).items():
        model = lgb.Booster(model_file=f"H:/teamSGBD/{name}.txt")

        # Load correct threshold for this model
        with open("H:/teamSGBD/model_results.csv") as fp:
            for row in fp:
                model_name, threshold, *_ = row.split(",")
                if name == model_name:
                    break

        # Create predictions
        y_prob = model.predict(
            df[cols].drop(columns=BASE), predict_disable_shape_check=True
        )
        y_pred = (y_prob > int(threshold) / 100).astype(int)
        predictions["prediction"] = y_pred

        # Save to path
        predictions.to_csv(f"H:/PREDICTIONS/{group_name}_{name}.csv")


def get_additional(train):
    personal_cols = [
        col
        for col in train.columns
        if not (any(col.endswith(network) for network in NETWORKS) or col in BASE)
    ]

    # Data set with only network features, no personal features
    cols = [
        col
        for col in train.columns
        if (any(col.endswith(network) for network in NETWORKS) or col in BASE)
    ]
    name = "model_nopersonal"
    additional = {name: cols}

    # Data sets with only one network (with and without personal features)
    for nopersonal in ["", "_nopersonal"]:
        for network in NETWORKS:
            # Select columns
            network_cols = [col for col in train.columns if network in col]
            if nopersonal:
                cols = BASE + network_cols
            else:
                cols = BASE + personal_cols + network_cols
            additional[f"model{network}{nopersonal}"] = cols


if __name__ == "__main__":
    main()
