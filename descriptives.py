import pandas as pd

def main():
    for nofeatures in ['', '_nofeatures']:
        for nonetwork in ['', '_nonetwork']:
            df = pd.read_csv(f'H:/teamSGBD/train{nofeatures}{nonetwork}.csv')
            df.groupby('children_post2021').describe().to_json(f'H:/teamSGBD/descriptives{nofeatures}{nonetwork}.json')
            df.describe().to_csv(f'H:/teamSGBD/descriptives{nofeatures}{nonetwork}.csv')

if __name__ == '__main__':
    main()
