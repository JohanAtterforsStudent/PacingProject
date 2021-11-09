import pandas as pd
# TODO: Add parameters of segments to calculate BP of. 
def BP(df, segments):
    df['BP'] = pd.to_timedelta(0)
    for i, seg in enumerate(segments):
        if i == 0:
            df['BP'] += (pd.to_timedelta(df[seg])) / 5
        else: 
            df['BP'] += (pd.to_timedelta(df[seg]) - pd.to_timedelta(df[segments[i - 1]])) / 5
    df['BP'] = df['BP'] / len(segments)
    # Old, not needed:
    # df['BP1'] = (pd.to_timedelta(df['_5Km']) / 5 + ((pd.to_timedelta(df['_10Km']) - pd.to_timedelta(df['_5Km'])) / 5)) / 2
    return df

def DoS(df, segments):
    allSegments = ['_5Km', '_10Km', '_15Km', '_20Km', 'FinishNetto']
    for i, seg in enumerate(segments):
        if seg == '_5Km':
            df['DoS_5Km'] = (pd.to_timedelta(df['_5Km']) / 5) / df['BP'] - 1
        elif i == 0:
            prevSegIndex = allSegments.index(seg) - 1
            df['DoS' + seg] = ((pd.to_timedelta(df[seg]) - (pd.to_timedelta(df[allSegments[prevSegIndex]])) ) / 5) / df['BP'] - 1
        else:
            df['DoS' + seg] = ((pd.to_timedelta(df[seg]) - pd.to_timedelta(df[segments[i-1]])) / 5) / df['BP'] - 1
    # Old, not needed
    #df['DoS_15'] = ((pd.to_timedelta(df['_15Km']) - pd.to_timedelta(df['_10Km'])) / 5) / df['BP'] - 1
    #df['DoS_20'] = ((pd.to_timedelta(df['_20Km']) - pd.to_timedelta(df['_15Km'])) / 5) / df['BP'] - 1
    #df['DoS_Finish'] = ((pd.to_timedelta(df['FinishNetto']) - pd.to_timedelta(df['_20Km'])) / 5) / df['BP'] - 1
    return df

def LoS(df, dos, segments):
    df['LoS'] = 0
    for seg in segments:
        if seg == 'DoSFinishNetto':
            df.loc[df['Dos' + seg] >= dos, 'LoS'] += 1.0975
        else: 
            df.loc[df['DoS' + seg] >= dos, 'LoS'] += 5
    # df.loc[df['DoS_15Km'] >= dos, 'LoS_'] += 5
    # df.loc[df['DoS_20Km'] >= dos, 'LoS_'] += 5
    # df.loc[df['DoSFinishNetto'] >= dos, 'LoS_'] += 1.0975
    return df