import pandas as pd
# Currently BP is average pace first 10 km. 
# TODO: Add parameters of segments to calculate BP of. 
def BP(df):
    df['BP'] = ((pd.to_timedelta(df['_5Km']) / 5) + (pd.to_timedelta(df['_10Km']) / 10)) / 2
    #df['BP'] = df['BP'] - pd.to_timedelta(df['BP'].dt.days, unit='d')
    #df['BP'] = df[cols].sum(axis=1)
    return df

def DoS(df):
    df['DoS_15'] = (pd.to_timedelta(df['_15Km']) / 15) / df['BP'] - 1
    df['DoS_20'] = (pd.to_timedelta(df['_20Km']) / 20) / df['BP'] - 1
    df['DoS_Finish'] = (pd.to_timedelta(df['FinishNetto']) / 21.0975) / df['BP'] - 1
    return df

def LoS(df, dos):
    df['LoS'] = 0
    df.loc[df['DoS_15'] >= dos, 'LoS'] += 5
    df.loc[df['DoS_20'] >= dos, 'LoS'] += 5
    return df