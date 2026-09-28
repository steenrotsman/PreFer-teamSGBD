#!/usr/bin/env python
# coding: utf-8

# ## Submission
# 1. Check if your code for preprocessing the data and training the model is reproducible (e.g. you set seed where necessary), readable, and clear (e.g. all the main steps commented briefly). Ideally, you should have a separate script for preprocessing the data (everything not included into a pipeline), for training and saving the model, and this script for producing predictions. 
# 
# 2. Edit the function below to produce predictions for the holdout data - H:/DATASETS/holdout_intermediate_leaderboard.csv 
# I.e. add preprocessing steps which are not included in the pipeline
# 
# 3. Apply this function to produce predictions for H:/DATASETS/holdout_intermediate_leaderboard.csv and save predictions in a csv file with your team name (H:/PREDICTIONS/{group_name}.csv). The csv file should have two columns, "RINPERSOON" (ID) and "prediction" (binary, 0 or 1)
# 
# There should be no missing values in the predictions.

# In[ ]:


# add all necessary packages here
import pandas as pd
# import joblib - forgot to add this to my environment, I use the lightgbm package to save and load the model
import lightgbm as lgb

from add_features import add_features, clean
from network import add_network

IDS = ['RINPERSOON', 'RINPERSOONS', 'RINPERSOONSHKW', 'RINPERSOONHKW', 'RINPERSOONSVERBINTENISP', 'RINPERSOONVERBINTENISP', 'SOORTOBJECTNUMMER', 'RINOBJECTNUMMER']

# edit this function
def predict_outcomes(path_to_holdout_data, group_name, full, train, nofeatures='', nonetwork=''):
    #Load the data (don't change this)
    df = pd.read_csv(path_to_holdout_data, low_memory=False, dtype={col: str for col in IDS})
    #save IDs to a new df
    predictions = df[["RINPERSOON"]]
    
    # Add our engineered features and network features, takes quite long
    if nofeatures == '_nofeatures':
        df = clean(df, train)
    else:
        df = clean(add_features(df), train)

    if nonetwork == '':
        df = add_network(df, full)
    
    #load model (ADD the right path to your model)
    model = lgb.Booster(model_file=f'H:/teamSGBD/model{nofeatures}{nonetwork}.txt')
    
    #load correct threshold for model
    with open('H:/teamSGBD/model_results.csv') as fp:
        for row in fp:
            name, threshold, *_ = row.split(',')
            if name == f'model{nofeatures}{nonetwork}':
                break

    #create predictions (don't change this)
    y_prob = model.predict(df.drop(columns=['RINPERSOON', 'children_post2021']), predict_disable_shape_check=True)
    y_pred = (y_prob > int(threshold)/100).astype(int)

    predictions["prediction"] = y_pred
    
    #save to path (don't change this)
    predictions.to_csv(f"H:/PREDICTIONS/{group_name}.csv", index=None)
    


# In[ ]:


# load reference data sets
full = pd.read_csv(f'H:/teamSGBD/full.csv', dtype={'RINPERSOON': str})
train_cols = pd.read_csv(f'H:/teamSGBD/train_nonetwork.csv', dtype={'RINPERSOON': str}, nrows=0).columns
train_nofeatures_cols = pd.read_csv(f'H:/teamSGBD/train_nofeatures_nonetwork.csv', dtype={'RINPERSOON': str}, nrows=0).columns


# In[ ]:


# produce predictions
predict_outcomes("H:/DATASETS/holdout_final_leaderboard.csv", "teamSGBD", full, train_cols)


# We have developed 4 models, with and without adding features to the provided training data and with and without network features. The main model that we submit for the leaderboard includes both our additional features and the network features. However, we would appreciate to get test set performances for the other three models as well.

# In[ ]:


path="H:/DATASETS/holdout_final_leaderboard.csv"
cols = train_cols
for nofeatures, cols in zip(['', '_nofeatures'], [train_cols, train_nofeatures_cols]):
    if nofeatures == '_nofeatures':
        full = pd.read_csv(f'H:/teamSGBD/full{nofeatures}.csv', dtype={'RINPERSOON': str})
    for nonetwork in ['', '_nonetwork']:
        if nofeatures == '' and nonetwork == '':
            continue
        predict_outcomes(path, f"teamSGBD{nofeatures}{nonetwork}", full, cols, nofeatures, nonetwork)

