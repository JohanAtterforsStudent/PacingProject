import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import DataCalc

def main():
    #ResultId;AthleteId;RaceId;RaceName;Gender;Birthyear;CountryIso;County;Municipality;Status;ActualStartTime;FinishNetto;Place;PlaceClass;PlaceTotal;_5Km;_10Km;_15Km;_20Km
    #rdf = pd.read_csv("Varvetresultat/samples.csv", header=0, sep=";")
    nan = 0
    for i in range(10):
        path = "Varvetresultat/201" + str(i) + ".csv"
        df = pd.read_csv(path, header=0, sep=";")
        #print(path)
        nan = (df['ResultId'].isna().sum())
        print(f"Year: 201{i}, NaN: {nan}")
        #print(df['Status'].value_counts(dropna=False))

    allSegments = ['_5Km', '_10Km', '_15Km', '_20Km', 'FinishNetto']
    BPSegments = allSegments[0:2]
    DoSSegments = allSegments[2:5]

    #DataCalc.BP(rdf, BPSegments)
    #DataCalc.DoS(rdf, DoSSegments)
    #DataCalc.LoS(rdf, 0.25, DoSSegments)
    #print("Length of slowdowns")
    #print(rdf['LoS'].value_counts(dropna=False))
    #print(rdf['Status'].value_counts(dropna=False))
    #print(rdf['Gender'].value_counts(dropna=False, normalize=True))

def scatter(dataFrame, xname, yname):
    x=dataFrame[xname]#.sample(n=200, random_state=1)
    y=dataFrame[yname]#.sample(n=200, random_state=1)
    plt.scatter(x, y, s=0.5)
    plt.xlabel(xname)
    plt.xlabel(yname)
    plt.show()

if __name__ == "__main__":
    main()