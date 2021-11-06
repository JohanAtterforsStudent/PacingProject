import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import DataCalc

def main():
    #ResultId;AthleteId;RaceId;RaceName;Gender;Birthyear;CountryIso;County;Municipality;Status;ActualStartTime;FinishNetto;Place;PlaceClass;PlaceTotal;_5Km;_10Km;_15Km;_20Km
    rdf = pd.read_csv("Varvetresultat/2019.csv", header=0, sep=";")
    #convToDate(rdf)
    
    # Remove columns 
    # ResultId = 0,
    # RaceId = 2, 
    # RaceName = 3, 
    # Birthyear = 5,
    # CountryIso = 6,
    # Municipality = 7,
    # Status = 8,
    # ActualStartTime = 10
    # Place = 12,
    # PlaceClass = 13,
    # PlaceTotal = 14
    rdf.drop(rdf.columns[[0,2,3,5,6,7,8,10,12,13,14]], axis=1, inplace=True)
    #print(rdf['Gender'].value_counts(dropna=False, normalize=True))
    #scatter(rdf, '_5Km', 'FinishNetto')
    DataCalc.BP(rdf)
    DataCalc.DoS(rdf)
    DataCalc.LoS(rdf, 0.25)
    print(rdf)
    print(rdf['LoS'].value_counts(dropna=False))
    print(rdf['Status'].value_counts(dropna=False))
    #print(rdf.info())
    #print(rdf.describe())

def scatter(dataFrame, xname, yname):
    x=dataFrame[xname]#.sample(n=200, random_state=1)
    y=dataFrame[yname]#.sample(n=200, random_state=1)
    plt.scatter(x, y, s=0.5)
    plt.xlabel(xname)
    plt.xlabel(yname)
    plt.show()
#def convToDate(rdf):
    #rdf['ActualStartTime'] = pd.to_datetime(rdf['ActualStartTime'], format=' %H:%M:%S').dt.time
    #rdf['FinishNetto'] = pd.to_datetime(rdf['FinishNetto'], format='%H:%M:%S').dt.time
    #rdf['_5Km'] = pd.to_datetime(rdf['_5Km'], format='%H:%M:%S').dt.time
    #rdf['_10Km'] = pd.to_datetime(rdf['_10Km'], format='%H:%M:%S').dt.time
    #rdf['_15Km'] = pd.to_datetime(rdf['_15Km'], format='%H:%M:%S').dt.time
    #rdf['_20Km'] = pd.to_datetime(rdf['_20Km'], format='%H:%M:%S').dt.time
    #pd.to_timedelta(rdf['ActualStartTime', 'FinishNetto'])

if __name__ == "__main__":
    main()