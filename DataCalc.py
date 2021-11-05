import pandas as pd
def BP(df, AthleteId):
    cols = ['_5Km', '_10Km']
    sum = 0
    for c in cols:
        print(df.loc[df['AthleteId'] == AthleteId, c])
        sum += df.loc[df['AthleteId'] == AthleteId, c]
    print(sum)