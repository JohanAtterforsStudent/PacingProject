import pandas as pd

def BP(df):
    df['BP'] = (pd.to_timedelta(df['_5Km']) + pd.to_timedelta(df['_10Km'])) / 10
    #df['BP'] = df[cols].sum(axis=1)
    return df