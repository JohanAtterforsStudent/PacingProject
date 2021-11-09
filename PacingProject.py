import pandas as pd
import numpy as np
import os
from os.path import exists

class PacingProject:
    def __init__(self):
        self.df = pd.DataFrame()
        self.directory = 'Varvetresultat'
        self.files = os.listdir(self.directory)
        self.timeColumns = ['Time','5km','10km','15km','20km']
        self.BPSegs = self.timeColumns[1:3]
        self.DoSSegs = [self.timeColumns[0], self.timeColumns[3:4]]

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
        print(self.df)
        #self.df.groupby(groupBy).apply(print)

    
    def BP(self):
        self.df['BP'] = 0
        for seg in self.BPSegs:
            self.df['BP'] += self.df[seg + "Pace"]
        self.df['BP'] = (self.df['BP'] / len(self.BPSegs)).astype(int)

    def DoS(self):
        for i, seg in enumerate(self.DoSSegs):
            if seg == '5Km':
                self.df['DoS_5Km'] = (pd.to_timedelta(df['_5Km']) / 5) / df['BP'] - 1
            elif i == 0:
                prevSegIndex = self.timeColumns.index(seg) - 1
                df['DoS' + seg] = ((pd.to_timedelta(df[seg]) - (pd.to_timedelta(df[self.allSegments[prevSegIndex]])) ) / 5) / df['BP'] - 1
            else:
                df['DoS' + seg] = ((pd.to_timedelta(df[seg]) - pd.to_timedelta(df[segments[i-1]])) / 5) / df['BP'] - 1
        return df

    def LoS(df, dos, segments):
        df['LoS'] = 0
        for seg in segments:
            if seg == 'DoSFinishNetto':
                df.loc[df['Dos' + seg] >= dos, 'LoS'] += 1.0975
            else: 
                df.loc[df['DoS' + seg] >= dos, 'LoS'] += 5
        return df

if __name__ == "__main__":
    PacingProject = PacingProject()
    PacingProject.MakeCSVs()       #Run too make a csv of all races in directory with renamed columns and a smaller with all runners that have completed all races
    PacingProject.ReadSmallTestCsv()
    PacingProject.AddPaces()
    PacingProject.BP()
    PacingProject.PrintStat('AthleteId')
