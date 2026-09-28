"""
Combine train.csv, data_supplementary.csv, and additional features from microdata into train.csv, eval.csv and full.csv;
train.csv is combined with network features and used as input for model training,
eval.csv is combined with network features and used as input for model evaluation,
full.csv is used to add network features.
"""

import logging
import re

import numpy as np
import pandas as pd
from pyreadstat._readstat_parser import PyreadstatError
from scipy.stats import linregress

IDS = ['RINPERSOON', 'RINPERSOONS', 'RINPERSOONSHKW', 'RINPERSOONHKW', 'RINPERSOONSVERBINTENISP', 'RINPERSOONVERBINTENISP', 'SOORTOBJECTNUMMER', 'RINOBJECTNUMMER']

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    # Load training and additional data sets
    df = pd.read_csv('H:/DATASETS/train.csv', dtype={col: str for col in IDS})
    additional = pd.read_csv('H:/DATASETS/data_supplementary.csv', dtype={col: str for col in IDS})
    df = pd.concat((df, additional)).drop_duplicates(subset='RINPERSOON')
    logger.info('Read provided data')

    # Add temporal features
    full = clean(add_features(df))
    full[full.evaluation_set==0].drop(columns='evaluation_set').to_csv('H:/teamSGBD/train_nonetwork.csv', index=False)
    full[full.evaluation_set==1].drop(columns='evaluation_set').to_csv('H:/teamSGBD/eval_nonetwork.csv', index=False)
    full.drop(columns=['evaluation_set', 'children_post2021']).to_csv('H:/teamSGBD/full.csv', index=False)
    logger.info('Saved added features splits')

    # Also save preprocessed data without added features
    df = clean(df)
    df[df.evaluation_set==0].drop(columns='evaluation_set').to_csv('H:/teamSGBD/train_nofeatures_nonetwork.csv', index=False)
    df[df.evaluation_set==1].drop(columns='evaluation_set').to_csv('H:/teamSGBD/eval_nofeatures_nonetwork.csv', index=False)
    df.drop(columns=['evaluation_set', 'children_post2021']).to_csv('H:/teamSGBD/full_nofeatures.csv', index=False)
    logger.info('Saved cleaned splits')



def add_features(df):
# 2016-2019 features
    df = df.rename(columns={'INHBRUTINKH': 'INHBRUTINKH_2020', 'OPLNIVSOI2021AGG4HBmetNIRWO': 'OPLNIVSOIHB_2020', 'VEHW1120ONRH': 'VEHW1120ONRH_2020'})
    start_year = 2016
    end_year = 2020
    for year in range(start_year, end_year):
        # Household income
        inhatab = pd.read_spss(f'G:/InkomenBestedingen/INHATAB/INHA{year}TABV2.sav', usecols=['RINPERSOONHKW', 'INHBRUTINKH'])
        inhatab = inhatab.rename(columns={'INHBRUTINKH': f'INHBRUTINKH_{year}'})
        inhatab[f'INHBRUTINKH_{year}'] = pd.to_numeric(inhatab[f'INHBRUTINKH_{year}'], errors='coerce')
        df = pd.merge(df, inhatab, on='RINPERSOONHKW', how='left')
        logger.info(f'Read {year} inhatab')
        # Attained education
        try:
            opleiding = pd.read_spss(f'G:/Onderwijs/HOOGSTEOPLTAB/{year}/HOOGSTEOPL{year}TABV3.sav', usecols=['RINPERSOON', 'OPLNIVSOI2021AGG4HBmetNIRWO', 'OPLNIVSOI2016AGG4HBMETNIRWO'])
        except PyreadstatError:
            opleiding = pd.read_spss(f'G:/Onderwijs/HOOGSTEOPLTAB/{year}/HOOGSTEOPL{year}TABV2.sav', usecols=['RINPERSOON', 'OPLNIVSOI2021AGG4HBmetNIRWO', 'OPLNIVSOI2016AGG4HBMETNIRWO'])
        opleiding = opleiding.iloc[:, [0, -1]].rename(columns={'OPLNIVSOI2021AGG4HBmetNIRWO': f'OPLNIVSOIHB_{year}', 'OPLNIVSOI2016AGG4HBMETNIRWO': f'OPLNIVSOIHB_{year}'})
        df = pd.merge(df, opleiding, on='RINPERSOON', how='left')
        logger.info(f'Read {year} opleiding')
        # Real estate assets; house worth
        try:
            vehtab = pd.read_spss(f'G:/InkomenBestedingen/VEHTAB/VEH{year}TABV3.sav', usecols=['RINPERSOONHKW', 'VEHW1120ONRH'])
        except PyreadstatError:
            vehtab = pd.read_spss(f'G:/InkomenBestedingen/VEHTAB/VEH{year}TABV4.sav', usecols=['RINPERSOONHKW', 'VEHW1120ONRH'])
        vehtab = vehtab.rename(columns={'VEHW1120ONRH': f'VEHW1120ONRH_{year}'})
        vehtab[f'VEHW1120ONRH_{year}'] = pd.to_numeric(vehtab[f'VEHW1120ONRH_{year}'], errors='coerce')
        df = pd.merge(df, vehtab, on='RINPERSOONHKW', how='left')
        logger.info(f'Read {year} real estate assets')

    # Slope of bruto income
    income = pd.melt(df, id_vars='RINPERSOONHKW', value_vars=[f'INHBRUTINKH_{year}' for year in range(start_year, 2021)]).dropna()
    income['variable'] = income.variable.str.replace('INHBRUTINKH_', '').astype('int64')
    slope = income.drop_duplicates().groupby('RINPERSOONHKW').apply(lambda v: linregress(v.variable, v.value)[0]).reset_index().rename(columns={0: 'INHBRUTINKH_slope'})
    df = pd.merge(df, slope, on='RINPERSOONHKW', how='left')
    logger.info('Created inhatab slope')

    # Slope of real estate assets
    assets = pd.melt(df, id_vars='RINPERSOONHKW', value_vars=[f'VEHW1120ONRH_{year}' for year in range(start_year, 2021)]).dropna()
    assets['variable'] = assets.variable.str.replace('VEHW1120ONRH_', '').astype('int64')
    slope = assets.drop_duplicates().groupby('RINPERSOONHKW').apply(lambda v: linregress(v.variable, v.value)[0]).reset_index().rename(columns={0: 'VEHW1120ONRH_slope'})
    df = pd.merge(df, slope, on='RINPERSOONHKW', how='left')
    logger.info('Created vehtab slope')

    # Last change of education
    df[f'OPLNIVSOIHB_lastchange'] = start_year
    for year in range(start_year, end_year):
        df.loc[df[f'OPLNIVSOIHB_{year}']!=df[f'OPLNIVSOIHB_{year+1}'], f'OPLNIVSOIHB_lastchange'] = year + 1
    logger.info(f'Created OPLNIVSOIHB_lastchange')

    # Was youngest child born in/after certain years?
    for year in [2007, 2015, 2018]:
        df[f'GEBJAARJONGSTEKINDHH_ge{year}'] = df['GEBJAARJONGSTEKINDHH'] >= year
        df.loc[df['GEBJAARJONGSTEKINDHH'].isna(), f'GEBJAARJONGSTEKINDHH_ge{year}'] = np.nan
        df[f'GEBJAARJONGSTEKINDHH_ge{year}'] = pd.to_numeric(df[f'GEBJAARJONGSTEKINDHH_ge{year}'])
        logger.info(f'Created GEBJAARJONGSTEKINDHH_ge{year}')

    # Year of last change in socioeconomic and civil status, address; recorded as yyyymmdd int, //10000 keeps only year
    for col in ['AANVSECM', 'AANVSECM_partner', 'AANVANGVERBINTENIS', 'GBADATUMAANVANGADRESHOUDING']:
        df[f'{col}_year'] = df[col] // 10000
    logger.info('Created year columns')

    # 2017 elections
    election = pd.read_csv('H:/DATASETS/Election results by individual.csv', dtype={'RINPERSOON': str}, usecols=lambda col: col.endswith('_gen2017') or col == 'RINPERSOON')
    df = df.merge(election, on='RINPERSOON', how='left')
    logger.info('Read election data')

    return df


def clean(df):
    # Convert some cols to numeric
    for col in ('GBAGEBOORTEJAARMOEDER', 'GBAGEBOORTEJAARVADER', 'GBAGEBOORTEJAARPARTNER'):
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert categorical vars to dummies
    df = df.drop(columns=IDS[1:])
    df = pd.get_dummies(df, columns=df.select_dtypes(('O', 'category')).columns.difference(['RINPERSOON']), drop_first=True)
    logger.info('One-hot encoded categories')

    # Rename columns without special characters
    df = df.rename(columns=lambda col: re.sub('[^\w\d_]+', '', col))

    return df

if __name__ == '__main__':
    main()
