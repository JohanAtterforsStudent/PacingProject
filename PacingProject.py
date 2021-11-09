import pandas as pd
import numpy as np
import os


class PacingProject:
    def __init__(self):
        self.df = pd.DataFrame()
        self.directory = 'Varvetresultat'
        self.files = os.listdir(self.directory)
        self.timeColumns = ['Time','5km','10km','15km','20km']

    def MakeCSVs(self):

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
        self.df.to_csv("AllResult.csv",index = False)


        #save everyone that has run all 10 races to smaller csv to use in testing
        self.df = self.df.groupby("AthleteId").filter(lambda x: len(x) > 9)
        self.df.to_csv("SmallTestResult.csv",index = False)


    def ReadLargeCsv(self): #read large file
        self.df = pd.read_csv('AllResult.csv')

    def ReadSmallTestCsv(self): #read small file
        self.df = pd.read_csv('SmallTestResult.csv')


    def AddPaces(self):
        self.df['5kmPace'] = int(self.df['5km']/5)
        self.df['10kmPace'] = int((self.df['10km'] - self.df['5km']) / 5)
        self.df['15kmPace'] = int((self.df['15km'] - self.df['10km']) / 5)
        self.df['20kmPace'] = int((self.df['20km'] - self.df['15km']) / 5)
        self.df['21kmPace'] = int((self.df['Time'] - self.df['20km']) / 1.0975)


    def PrintStat(self,groupBy,):
        pass

if __name__ == "__main__":
    PacingProject = PacingProject()
    PacingProject.MakeCSVs()       #Run too make a csv of all races in directory with renamed columns and a smaller with all runners that have completed all races
    PacingProject.ReadSmallTestCsv()
    PacingProject.AddPaces()
