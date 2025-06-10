import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler


def deal_category (df: pd.DataFrame):
    for i in df.columns:
        if df[i].dtype == 'category':
            df[i] = pd.to_numeric(df[i], errors='coerce')
    return df

def preprocess ():
    df = pd.read_parquet('input/SIMORTN5.parquet.gzip')

    df = deal_category(df)

    df = df.loc[(df['RACACOR'] != 2) & (df['RACACOR'] != 5)]

    df = df.dropna(axis=1)

    scale = StandardScaler()
    X = pd.DataFrame(scale.fit_transform(df.drop(['OBITO'], axis=1), ))
    y = df['OBITO']

    # X_train, X_test, y_train, y_test
    return tts(X, y, test_size=0.7, random_state=72)