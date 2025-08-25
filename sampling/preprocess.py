import pandas as pd


def deal_category(df: pd.DataFrame):
    for i in df.columns:
        if df[i].dtype == 'category':
            df[i] = pd.to_numeric(df[i], errors='coerce')
    return df


def apply_preprocess():
    df = pd.read_parquet('data/raw/SIMORTN5.parquet.gzip')

    df = deal_category(df)

    df = df.loc[(df['RACACOR'] != 2) &
                (df['RACACOR'] != 5)]

    df = df.dropna()

    df.set_index('index', inplace=True)

    df.to_parquet('data/preprocessed/data.parquet.gzip')

    return df


def preprocess():
    try:
        df = pd.read_parquet('data/preprocessed/' +
                             'data.parquet.gzip')
    except FileNotFoundError:
        df = apply_preprocess()

    return df[:500_000]
