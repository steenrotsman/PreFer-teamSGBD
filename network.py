"""
Load train and additional data. For every observation in the train set, find the ids of people several networks.
For every network, aggregate feature values for neighbors and add as features to training data.
"""
import logging

import pandas as pd

BUURTGENOOT = 102
SIBLING_FULL = 306
SIBLING_HALF = 307
SIBLING_UNKNOWN = 308
COUSIN = 309
SIBLING_LAW = 316
STEPSIBLING = 319
STEPCOUSIN = 320
BIRTHYEAR_45 = 2020 - 45
NEIGHBORHOOD_INCLUDE = [BUURTGENOOT]
FAMILY_INCLUDE = [SIBLING_FULL, SIBLING_HALF, STEPSIBLING, SIBLING_UNKNOWN, COUSIN]

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    for nofeatures in ['', '_nofeatures']:
        train = pd.read_csv(f'H:/teamSGBD/train{nofeatures}_nonetwork.csv', dtype={'RINPERSOON': str})
        logger.info(f'Read train{nofeatures}_nonetwork.csv')
        eval = pd.read_csv(f'H:/teamSGBD/eval{nofeatures}_nonetwork.csv', dtype={'RINPERSOON': str})
        logger.info(f'Read eval{nofeatures}_nonetwork.csv')
        full = pd.read_csv(f'H:/teamSGBD/full{nofeatures}.csv', dtype={'RINPERSOON': str})
        logger.info(f'Read full{nofeatures}.csv')

        # Convert dtypes to 32-bit to save space
        for dtype in ['int', 'float']:
            for df in [train, eval, full]:
                df[df.select_dtypes(f'{dtype}64').columns] = df.select_dtypes(f'{dtype}64').astype(f'{dtype}32')
        logger.info('Converted dtypes to 32 bit')
            
        add_network(train, full).to_csv(f'H:/teamSGBD/train_network{nofeatures}.csv', index=False)
        logger.info(f'Saved train_network{nofeatures}.csv')
        add_network(eval, full).to_csv(f'H:/teamSGBD/eval_network{nofeatures}.csv', index=False)
        logger.info(f'Saved eval_network{nofeatures}.csv')

def add_network(df, full):
    # Buren and familie networks
    for group, include in zip(['BUREN', 'FAMILIE'], [NEIGHBORHOOD_INCLUDE, FAMILY_INCLUDE]):
        path = f'G:/Bevolking/{group}NETWERKTAB/{group}NETWERK2020TABV1.csv'
        usecols = ['RINPERSOON', 'RINPERSOONRELATIE', 'RELATIE']
        chunksize = 5 * 10**7
        chunks = []
        for i, chunk in enumerate(pd.read_csv(path, dtype={'RINPERSOON': str, 'RINPERSOONRELATIE': str}, sep=';', usecols=usecols, chunksize=chunksize)):
            logger.info(f'Read {group} chunk {i}')
            chunks.append(chunk[
                # Use only relationships defined above
                (chunk.RELATIE.isin(include)) 
                # We don't need the rows where RINPERSOON is not in df
                & (chunk.RINPERSOON.isin(df.RINPERSOON))
                # We can't get the values where RINPERSOONRELATIE is not in df_additional
                & (chunk.RINPERSOONRELATIE.isin(full.RINPERSOON))
            ]
            .drop(columns='RELATIE')
            )
        network = (pd.concat(chunks))
    
        # Merging all rows at once won't fit in memory: process in chunks based on last digit of RINPERSOON
        network_features = []
        buckets = network.RINPERSOON.astype('int32') % 10
        for i in range(10):
            network_features.append(
                network[buckets == i]
                .merge(full, left_on='RINPERSOONRELATIE', right_on='RINPERSOON', suffixes=[None, f'_'])
                .drop(columns=['RINPERSOON_', 'RINPERSOONRELATIE'])
                .groupby('RINPERSOON', as_index=False)
                .mean()
            )
            logger.info(f'Added {group} features bucket {i}')

        df = df.merge(pd.concat(network_features), how='left', on='RINPERSOON', suffixes=[None, f'_{group}'])
        logger.info(f'Merged {group} features')

    # Collega network
    path = f'G:/Bevolking/COLLEGANETWERKTAB/COLLEGANETWERK2020TABV1.csv'
    usecols = ['RINPERSOON', 'RINPERSOONRELATIE']
    chunks = []
    for i, chunk in enumerate(pd.read_csv(path, dtype={'RINPERSOON': str, 'RINPERSOONRELATIE': str}, sep=';', usecols=usecols, chunksize=chunksize)):
        logger.info(f'Read COLLEGA chunk {i}')
        chunk = chunk.merge(full.loc[:, ['RINPERSOON', 'GBAGEBOORTEJAAR']], on='RINPERSOON')
        chunks.append(chunk[
            # Filter out if too old (>45)
            (chunk.GBAGEBOORTEJAAR >= BIRTHYEAR_45)  
            # We don't need the rows where RINPERSOON is not in df
            & (chunk.RINPERSOON.isin(df.RINPERSOON))
            # We can't get the values where RINPERSOONRELATIE is not in df_additional
            & (chunk.RINPERSOONRELATIE.isin(full.RINPERSOON))
        ]
        .drop(columns='GBAGEBOORTEJAAR')
        )
    network = (pd.concat(chunks))

    network_features = []
    buckets = network.RINPERSOON.astype('int32') % 10
    for i in range(10):
        network_features.append(
            network[buckets == i]
            .merge(full, left_on='RINPERSOONRELATIE', right_on='RINPERSOON', suffixes=[None, f'_'])
            .drop(columns=['RINPERSOON_', 'RINPERSOONRELATIE'])
            .groupby('RINPERSOON', as_index=False)
            .mean()
        )
        logger.info(f'Added COLLEGA features bucket {i}')

    df = df.merge(pd.concat(network_features), how='left', on='RINPERSOON', suffixes=[None, f'_COLLEGA'])
    logger.info(f'Merged COLLEGA features')
    return df

if __name__ == '__main__':
    main()