from matplotlib.pyplot import colorbar, legend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import exists

class PacingProject:
    def __init__(self):
        self.df = pd.DataFrame()
        self.directory = 'Varvetresultat'
        self.files = os.listdir(self.directory)
        self.timeColumns = ['Time','5km','10km','15km','20km']
        # 5Km, 10Km
        self.BPSegs = self.timeColumns[1:3]
        # 15Km, 20Km, Time
        self.DoSSegs = self.timeColumns[3:5] + [self.timeColumns[0]]

    def MakeCSVs(self):
        if(not exists("Varvetresultat/AllResult.csv")):
            #read and merge all results csvs
            for file in self.files:
                temp = pd.read_csv(self.directory+'/'+file, header=0, sep=";")
                #drop not used cols
                temp.drop(['ResultId','RaceId','CountryIso','County','Municipality','ActualStartTime','Place','PlaceClass','PlaceTotal'], axis=1, inplace=True)
                #drop not a number
                temp = temp.dropna()
                self.df = pd.concat([self.df, temp],ignore_index=True)
            #rename to easier names
            self.df = self.df.rename(columns={"RaceName":"Year","FinishNetto":"Time","_5Km":"5km","_10Km":"10km","_15Km": "15km", "_20Km": "20km"})

            #transform timestring to int of with number of seconds
            for timeColumn in self.timeColumns:
                self.df[timeColumn] = pd.to_timedelta(self.df[timeColumn])
                self.df[timeColumn] = self.df[timeColumn].dt.seconds

            #transform datatypes to int64 to avoid storing as float
            self.df['Year'] = self.df['Year'].str[-4:].astype(int)
            self.df['Year'] = self.df['Year'].astype("Int64")
            self.df['AthleteId'] = self.df['AthleteId'].astype("Int64")
            self.df['Birthyear'] = self.df['Birthyear'].astype("Int64")

            #save all results in file
            self.df.to_csv(self.directory + "/AllResult.csv",index = False)


            #save everyone that has run all 10 races to smaller csv to use in testing
            self.df = self.df.groupby("AthleteId").filter(lambda x: len(x) > 9)
            self.df.to_csv(self.directory + "/SmallTestResult.csv",index = False)


    def ReadLargeCsv(self): #read large file
        self.df = pd.read_csv(self.directory + '/AllResult.csv')

    def ReadSmallTestCsv(self): #read small file
        self.df = pd.read_csv(self.directory + '/SmallTestResult.csv')


    def AddPaces(self):
        self.df['5kmPace'] = (self.df['5km']/5).astype(int)
        self.df['10kmPace'] = ((self.df['10km'] - self.df['5km']) / 5).astype(int)
        self.df['15kmPace'] = ((self.df['15km'] - self.df['10km']) / 5).astype(int)
        self.df['20kmPace'] = ((self.df['20km'] - self.df['15km']) / 5).astype(int)
        self.df['21kmPace'] = ((self.df['Time'] - self.df['20km']) / 1.0975).astype(int)


    def PrintStat(self,groupBy):
        self.df.groupby(groupBy).apply(print)

    
    def BP(self):
        self.df['BP'] = 0
        for seg in self.BPSegs:
            self.df['BP'] += self.df[seg + "Pace"]
        self.df['BP'] = (self.df['BP'] / len(self.BPSegs)).astype(int)

    def DoS(self):
        for seg in self.DoSSegs:
            if seg == "Time":
                self.df['DoS' + seg] = self.df["21kmPace"] / self.df['BP'] - 1
            else :
                self.df['DoS' + seg] = self.df[seg + "Pace"] / self.df['BP'] - 1

    def LoS(self, dos):
        self.df['LoS'] = 0
        for seg in self.DoSSegs:
            if seg == 'Time':
                self.df.loc[self.df['DoS' + seg] >= dos, 'LoS'] += 1.0975
            else: 
                self.df.loc[self.df['DoS' + seg] >= dos, 'LoS'] += 5
    
    def SensitivityPlot(self):
        df = pd.DataFrame()
        df["SlowdownThresholds"] = pd.Series(np.arange(0,0.80, 0.05))

        df["5Km"] = 0
        df["10Km"] = 0
        df["15Km"] = 0
        df["20Km"] = 0
        df["21Km"] = 0

        for i, value in enumerate(df["SlowdownThresholds"]):
            self.LoS(value)
            for j, value in enumerate([5,10,15,20,21.0975]):
                df.iat[i,j+1] = self.df.loc[(self.df["LoS"] >= value), ["AthleteId"]].count() / self.df["AthleteId"].count()

        x=df["SlowdownThresholds"]
        plt.plot(x, df["5Km"], 'go--', label="5Km")
        plt.plot(x, df["10Km"],'bD--', label="10Km")
        plt.plot(x, df["15Km"],'yx--', label="15Km")
        plt.legend()
        #plt.plot(x, df["21Km"], color='r')
        plt.xlabel("Slowdown Thresholds")
        plt.ylabel("Proportion")
        plt.show()


if __name__ == "__main__":
    PacingProject = PacingProject()
    PacingProject.MakeCSVs()       #Run too make a csv of all races in directory with renamed columns and a smaller with all runners that have completed all races
    PacingProject.ReadSmallTestCsv()
    PacingProject.AddPaces()
    PacingProject.BP()
    PacingProject.DoS()
    PacingProject.SensitivityPlot()
    #PacingProject.PrintStat('AthleteId')
