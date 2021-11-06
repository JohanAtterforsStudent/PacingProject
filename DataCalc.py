import pandas as pd
# Currently BP is average pace first 10 km. 
# TODO: Add parameters of segments to calculate BP of. 
def BP(df):
    df['BP'] = (pd.to_timedelta(df['_5Km']) / 5 + ((pd.to_timedelta(df['_10Km']) - pd.to_timedelta(df['_5Km'])) / 5)) / 2
    return df

def DoS(df):
    df['DoS_15'] = ((pd.to_timedelta(df['_15Km']) - pd.to_timedelta(df['_10Km'])) / 5) / df['BP'] - 1
    df['DoS_20'] = ((pd.to_timedelta(df['_20Km']) - pd.to_timedelta(df['_15Km'])) / 5) / df['BP'] - 1
    df['DoS_Finish'] = ((pd.to_timedelta(df['FinishNetto']) - pd.to_timedelta(df['_20Km'])) / 5) / df['BP'] - 1
    return df

def LoS(df, dos):
    df['LoS'] = 0
    df.loc[df['DoS_15'] >= dos, 'LoS'] += 5
    df.loc[df['DoS_20'] >= dos, 'LoS'] += 5
    df.loc[df['DoS_Finish'] >= dos, 'LoS'] += 1.0975
    return df