import pandas as pd
import numpy as np
import os

class PacingProject:
    def __init__(self):
        self.df = pd.DataFrame()
        self.directory = '/content/drive/MyDrive/Varvetresultat'
        self.files = os.listdir(self.directory)
        self.files.remove('.ipynb_checkpoints')
        self.files.remove('Models')
        self.files.remove('Figs')
        self.files.remove('SmallTestResult.csv')
        self.files.remove('AllResult.csv')
        self.timeColumns = ['Time', '5km', '10km', '15km', '20km', 'ActualStartTime']
        # 5Km, 10Km
        self.BPSegs = self.timeColumns[1:3]
        # 15Km, 20Km, Time
        self.DoSSegs = self.timeColumns[3:5] + [self.timeColumns[0]]
        self.relativePaces = ['5KmRelativePace', '10KmRelativePace', '15KmRelativePace', '20KmRelativePace',
                              '21KmRelativePace']
        self.boxLimits = [0.98, 1.02]
        self.boxLimits.append(10)


    def MakeCSVs(self):
        if (os.path.isfile("/content/drive/MyDrive/Varvetresultat/AllResult.csv")):
            # read and merge all results csvs
            sum_nans = 0
            for file in self.files:
              temp = pd.read_csv(self.directory + "/" + file, header=0, sep=";")
              # drop not used cols
              temp.drop(["ResultId", 'RaceId', 'CountryIso', 'County', 'Municipality', 'Place', 'PlaceClass', 'PlaceTotal'], axis=1, inplace=True)
              # drop not a number
              nans = temp.isnull().any(axis=1).sum()
              #print(f"Nr rows dropped: {nans}")
              sum_nans += nans

              temp = temp.dropna()
              self.df = pd.concat([self.df, temp], ignore_index=True)
            print(f"Total nans dropped: {sum_nans}")
            # rename to easier names
            self.df = self.df.rename(
                columns={"RaceName": "Year", "FinishNetto": "Time", "_5Km": "5km", "_10Km": "10km", "_15Km": "15km",
                         "_20Km": "20km"})

            # transform timestring to int of with number of seconds
            for timeColumn in self.timeColumns:
                self.df[timeColumn] = pd.to_timedelta(self.df[timeColumn])
                self.df[timeColumn] = self.df[timeColumn].dt.seconds

            # transform datatypes to int64 to avoid storing as float
            self.df['Year'] = self.df['Year'].str[-4:].astype(int)
            self.df['Year'] = self.df['Year'].astype("Int64")
            self.df['AthleteId'] = self.df['AthleteId'].astype("Int64")
            self.df['Birthyear'] = self.df['Birthyear'].astype("Int64")

            # save all results in file
            self.df.to_csv(self.directory + "/AllResult.csv", index=False)

            # save everyone that has run all 10 races to smaller csv to use in testing
            self.df = self.df.groupby("AthleteId").filter(lambda x: len(x) > 9)
            self.df.to_csv(self.directory + "/SmallTestResult.csv", index=False)

    def ReadLargeCsv(self):  # read large file
        self.df = pd.read_csv(self.directory + '/AllResult.csv')
        self.AddPaces()
        #self.AddHistory()
        #self.LearnFromPrevious()
        #self.AddAgeExp()

    def ReadSmallTestCsv(self):  # read small file
        self.df = pd.read_csv(self.directory + '/SmallTestResult.csv')
        self.AddPaces()
        self.AddHistory()
        self.LearnFromPrevious()
        self.AddPB()
    
    def AddAgeExp(self):
      self.df['Age'] = self.df['Year'] - self.df['Birthyear']
      self.df['Runs'] = self.df.groupby('AthleteId').cumcount()+1

    def AddPaces(self):
        self.df['5kmPace'] = (self.df['5km'] / 5)
        self.df['10kmPace'] = ((self.df['10km'] - self.df['5km']) / 5)
        self.df['15kmPace'] = ((self.df['15km'] - self.df['10km']) / 5)
        self.df['20kmPace'] = ((self.df['20km'] - self.df['15km']) / 5)
        self.df['21kmPace'] = ((self.df['Time'] - self.df['20km']) / 1.0975)
        self.df['TotalPace'] = (self.df['Time'] / 21.0975)

        self.df['5KmRelativePace'] = self.df['5kmPace'] / self.df['TotalPace']
        self.df['10KmRelativePace'] = self.df['10kmPace'] / self.df['TotalPace']
        self.df['15KmRelativePace'] = self.df['15kmPace'] / self.df['TotalPace']
        self.df['20KmRelativePace'] = self.df['20kmPace'] / self.df['TotalPace']
        self.df['21KmRelativePace'] = self.df['21kmPace'] / self.df['TotalPace']

        self.df['10KmRP'] = self.df['10kmPace'] / self.df['5kmPace']
        self.df['15KmRP'] = self.df['15kmPace'] / self.df['10kmPace']
        self.df['20KmRP'] = self.df['20kmPace'] / self.df['15kmPace']
        self.df['21KmRP'] = self.df['21kmPace'] / self.df['20kmPace']

        self.df['FirstHalfPace'] = (self.df['10km'] / 10)
        self.df['SecondHalfPace'] = ((self.df['Time']-self.df['10km']) / 11.0975)

        self.df['SplitRatio'] = self.df['SecondHalfPace']/self.df['FirstHalfPace']
        self.df['SplitRatioTime'] = self.df['Time']-self.df['10km']*2.10975

        self.BP()
        self.DoS()
        self.AddWeather()

    def AddHistory(self):
        self.df['LastTime'] = self.df.groupby(['AthleteId'])['Time'].shift(-1)
        self.df['LastSplitRatio'] = self.df.groupby(['AthleteId'])['SplitRatio'].shift(-1)
        self.df = self.df.dropna()
    
    def LearnFromPrevious(self):

        #< 0.95 = negative split, 0.95-1.05 = even, 1,05-1.15 = positive, 1.15 or more = htw
        conditions = [
            (self.df['SplitRatio'] <= 0.95),
            (self.df['SplitRatio'] > 0.95) & (self.df['SplitRatio'] <= 1.05),
            (self.df['SplitRatio'] > 1.05) & (self.df['SplitRatio'] <= 1.15),
            (self.df['SplitRatio'] > 1.15)
        ]
        conditions2 = [
            (self.df['LastSplitRatio'] <= 0.95),
            (self.df['LastSplitRatio'] > 0.95) & (self.df['LastSplitRatio'] <= 1.05),
            (self.df['LastSplitRatio'] > 1.05) & (self.df['LastSplitRatio'] <= 1.15),
            (self.df['LastSplitRatio'] > 1.15)
        ]

        # create a list of the values we want to assign for each condition
        values = ['NEG', 'EVEN', 'POS', 'HTW']

        # create a new column and use np.select to assign values to it using our lists as arguments
        self.df['PaceGroup'] = np.select(conditions, values)
        self.df['LastPaceGroup'] = np.select(conditions2, values)

        self.df['count'] = 1

        #print(self.df.head(10))


        groups = self.df.groupby(['PaceGroup', 'LastPaceGroup'])
        counts = {i[0]: (len(i[1])) for i in groups}
        #print(counts)

        matrix = pd.DataFrame()

        for x in values:
            matrix[str(x)] = pd.Series([counts.get((x, y), 0) for y in values], index=values)

        #print(matrix)

        row_sums = matrix.sum(axis=1)
        new_matrix = matrix / row_sums[:, np.newaxis]

        #print(new_matrix)

    def AddPB(self):
        min_values = self.df.groupby('AthleteId').agg({'Time': np.min})
        self.df = self.df.merge(min_values, how='outer', on='AthleteId')
        self.df.rename(columns={'Time_x': 'Time', 'Time_y': 'Pb'}, inplace=True)

    def BP(self):
        self.df['BP'] = 0
        for seg in self.BPSegs:
            self.df['BP'] += self.df[seg + "Pace"]
        self.df['BP'] = (self.df['BP'] / len(self.BPSegs))

    def DoS(self):
        for seg in self.DoSSegs:
            if seg == "Time":
                self.df['DoS' + seg] = self.df["21kmPace"] / self.df['BP'] - 1
            else:
                self.df['DoS' + seg] = self.df[seg + "Pace"] / self.df['BP'] - 1

    def LoS(self, dos):
        self.df['LoS'] = 0
        for seg in self.DoSSegs:
            if seg == 'Time':
                self.df.loc[self.df['DoS' + seg] >= dos, 'LoS'] += 1.0975
            else:
                self.df.loc[self.df['DoS' + seg] >= dos, 'LoS'] += 5

    def RemoveFaultyData(self):

        print('Removing data containg errors')

        pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.expand_frame_repr', False)
        # print(str((self.df['Time'] == 0).sum())+' runners with 00:00:00 as finish time')
        # print(self.df[self.df['Time'] == 0])
        pre = len(self.df)
        #print(f"Length df before removing stuff: {pre}")
        bad_time = sum(self.df['Time'] == 0)
        bad_paces = 0
        bad_paces += sum(self.df['21KmRelativePace'] >= 2)
        bad_paces += sum(self.df['20KmRelativePace'] >= 2)
        bad_paces += sum(self.df['15KmRelativePace'] >= 2)
        bad_paces += sum(self.df['10KmRelativePace'] >= 2)
        bad_paces += sum(self.df['5KmRelativePace'] >= 2)
        #print(f"Runners with time == 0: {bad_time}")
        #print(f"Runners with RelativePace > 2: {bad_paces}")
        bad_id = sum(self.df["AthleteId"] == 0)
        #print(f"Runners with bad id > 2: {bad_id}")

        self.df = self.df[self.df['Time'] != 0]

        self.df = self.df[self.df['21KmRelativePace'] < 2]
        self.df = self.df[self.df['20KmRelativePace'] < 2]
        self.df = self.df[self.df['15KmRelativePace'] < 2]
        self.df = self.df[self.df['10KmRelativePace'] < 2]
        self.df = self.df[self.df['5KmRelativePace'] < 2]
        self.df = self.df[self.df["AthleteId"] != 0]

        print(f"Runners removed due to error: {pre - len(self.df)}")

    def AddWeather(self):
        '''22 maj 2010
        21 maj 2011
        12 maj 2012
        18 maj 2013
        17 maj 2014
        23 maj 2015
        21 maj 2016
        20 maj 2017
        19 maj 2018
        18 maj 2019'''
        years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
        dewPoints = [16.4,8.6,3.4,18,10.0,6.0,11.3,10.7,10.5,13.6]
        temperatures = [21.7,16.6,13.6,25.0,18.9,14.7,15.1,13.9,20.0,19.4]
        relativeHumidity = [55,54,44,50,56,52,78,75,49,64]

        dewPointDict = {}
        temperatureDict = {}
        relativeHumidityDict = {}

        for year,dewPoint,temperature,relativeHum in zip(years,dewPoints,temperatures,relativeHumidity):

            dewPointDict[year] = dewPoint
            temperatureDict[year] = temperature
            relativeHumidityDict[year] = relativeHum

        self.df['dewPoint'] = self.df['Year'].map(dewPointDict)
        self.df['temperature'] = self.df['Year'].map(temperatureDict)
        self.df['relativeHumidity'] = self.df['Year'].map(relativeHumidityDict)
