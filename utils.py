import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path, rows=None):

    if rows:
        df = pd.read_csv(path, nrows=rows)
    else:
        df = pd.read_csv(path)

    return df


def preprocess(df):

    # remove account IDs
    df = df.drop(["nameOrig","nameDest"], axis=1)

    # encode transaction type
    encoder = LabelEncoder()
    df["type"] = encoder.fit_transform(df["type"])

    return df