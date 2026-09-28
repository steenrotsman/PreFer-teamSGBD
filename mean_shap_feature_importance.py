import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from add_features import IDS, add_features, clean
from network import add_network

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

path_to_holdout_data = ""


train = pd.read_csv("H:/teamSGBD/train.csv", dtype={"RINPERSOON": str})
full = pd.read_csv("H:/teamSGBD/full.csv", dtype={"RINPERSOON": str})
df = pd.read_csv(
    path_to_holdout_data, low_memory=False, dtype={col: str for col in IDS}
)
df = add_network(clean(add_features(df)), full)
model = lgb.Booster(model_file="H:/teamSGBD/model.txt")
expected_features = model.feature_name()
df_aligned = df.reindex(columns=expected_features, fill_value=0)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_aligned)

# calculate mean shap value per feature
mean_shap = pd.Series(np.mean(np.abs(shap_values), axis=0), index=df_aligned.columns)
mean_shap = mean_shap.sort_values(ascending=False)
mean_shap.to_csv('feature_importance.csv', header = ['mean_shap'])
