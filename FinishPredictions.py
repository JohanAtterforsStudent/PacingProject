import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rcParams['figure.constrained_layout.use'] = True
from matplotlib.pyplot import colorbar, legend
import os
from os.path import exists
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import matplotlib.dates as md



def minSecString(val):
    val = int(val)
    if val < 60:
        return str(val)
    elif val % 60 > 9:
        return str(int(val // 60)) + ':' + str(int(val % 60))
    else:
        return str(int(val // 60)) + ':0' + str(int(val % 60))

def hourMinString(val):
    val = val/60
    val = int(val)
    if val < 60:
        return str(val)
    elif val % 60 > 9:
        return str(int(val // 60)) + ':' + str(int(val % 60))
    else:
        return str(int(val // 60)) + ':0' + str(int(val % 60))

class PacingProject:
    def __init__(self):
        self.df = pd.DataFrame()
        self.directory = 'Varvetresultat'
        self.files = os.listdir(self.directory)
        self.timeColumns = ['Time', '5km', '10km', '15km', '20km']
        # 5Km, 10Km
        self.BPSegs = self.timeColumns[1:3]
        # 15Km, 20Km, Time
        self.DoSSegs = self.timeColumns[3:5] + [self.timeColumns[0]]
        self.relativePaces = ['5KmRelativePace', '10KmRelativePace', '15KmRelativePace', '20KmRelativePace',
                              '21KmRelativePace']
        self.boxLimits = [0.98, 1.02]
        self.boxLimits.append(10)

    def MakeCSVs(self):
        if (not exists("Varvetresultat/AllResult.csv")):
            # read and merge all results csvs
            for file in self.files:
                temp = pd.read_csv(self.directory + '/' + file, header=0, sep=";")
                # drop not used cols
                temp.drop(['ResultId', 'RaceId', 'CountryIso', 'County', 'Municipality', 'ActualStartTime', 'Place',
                           'PlaceClass', 'PlaceTotal'], axis=1, inplace=True)
                # drop not a number
                temp = temp.dropna()
                self.df = pd.concat([self.df, temp], ignore_index=True)
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

    def ReadCsv(self):
        self.df = pd.read_csv(self.directory + '/AllResult.csv')
        self.AddPaces()
        self.AddWeather()
        self.AddAge()
        self.AddHistory()

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

        self.df['FirstHalfPace'] = (self.df['10km'] / 10)
        self.df['SecondHalfPace'] = ((self.df['Time']-self.df['10km']) / 11.0975)
        self.df['SplitDifference'] = self.df['Time']-self.df['10km']*2.10975

    def AddAge(self):
        self.df['Age'] = self.df['Year']-self.df['Birthyear']

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
        Temperatures = [21.7,16.6,13.6,25.0,18.9,14.7,15.1,13.9,20.0,19.4]



        TemperatureDict = {}



        for year,Temperature, in zip(years,Temperatures):

            TemperatureDict[year] = Temperature

        self.df['Temperature'] = self.df['Year'].map(TemperatureDict)


    def AddHistory(self):

        self.df['LastYearTime'] = self.df.groupby(['AthleteId'])['Time'].shift(1)
        self.df['LastYearSplitDifference'] = self.df.groupby(['AthleteId'])['SplitDifference'].shift(1)
        self.df = self.df[self.df['AthleteId'] != 0]


        mask = self.df['Year'] == 2019
        self.data2019 = self.df[mask].copy()

    def RemoveFaultyData(self):

        print('Removing data containg errors')
        pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.expand_frame_repr', False)


        self.df = self.df[self.df['Time'] != 0]
        self.df = self.df[self.df['21KmRelativePace'] < 2]
        self.df = self.df[self.df['20KmRelativePace'] < 2]
        self.df = self.df[self.df['15KmRelativePace'] < 2]
        self.df = self.df[self.df['10KmRelativePace'] < 2]
        self.df = self.df[self.df['5KmRelativePace'] < 2]
        self.df = self.df[self.df['AthleteId'] != 0]

    def PredictMethodComparison(self):

        print('Datapoints used in features ANN')
        print(len(self.df))

        plt.clf()

        BLResult = []
        LRResult = []
        ANNResult = []


        splits = ['5km', '10km', '15km', '20km']

        usedFeatures = []
        numberOfFeatures = len(usedFeatures)

        usedColumns = usedFeatures

        for split in splits:

            usedColumns.append(split)

            y = self.df['Time']
            X = self.df[usedColumns]

            y = np.array(y).reshape(-1, 1)
            X = np.array(X)

            kf =  KFold(n_splits=5)

            BLErrorCrossValIter = []
            LRErrorCrossValIter = []
            ANNErrorCrossValIter = []

            for train_index, test_index in kf.split(X):

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

                nPassedSplits = len(usedColumns)-numberOfFeatures
                if nPassedSplits == 1:
                    lastPace = X_test[:, numberOfFeatures] / 5
                else:
                    lastPace = (X_test[:, nPassedSplits+numberOfFeatures-1] - X_test[:, nPassedSplits + numberOfFeatures-2]) / 5


                baselinePrediction = X_test[:, nPassedSplits+numberOfFeatures-1] + lastPace * (1.0975 + (4 - nPassedSplits) * 5)

                xScaler = StandardScaler()
                yScaler = StandardScaler()

                X_train = xScaler.fit_transform(X_train)
                X_test = xScaler.transform(X_test)


                y_train = yScaler.fit_transform(y_train)
                y_test = yScaler.transform(y_test)

                # LR
                LRmodel = LinearRegression()
                LRmodel.fit(X_train, y_train)

                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


                cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
                ANNmodel = tf.keras.models.Sequential()
                ANNmodel.add(tf.keras.layers.Dense(units=40, activation='relu'))

                ANNmodel.add(tf.keras.layers.Dense(units=1))
                ANNmodel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics='mae')
                history = ANNmodel.fit(X_train, y_train, batch_size=1024,
                                           epochs=2000, callbacks=cb, validation_data=(X_val, y_val))


                ANNPrediction = pd.DataFrame(yScaler.inverse_transform(ANNmodel.predict(X_test)))


                y_test = yScaler.inverse_transform(y_test).reshape(len(y_test))
                linearRegressionPrediction = yScaler.inverse_transform(LRmodel.predict(X_test)).reshape(len(y_test))
                ANNPrediction = yScaler.inverse_transform(ANNmodel.predict(X_test)).reshape(len(y_test))


                BLError = np.mean(abs(baselinePrediction - y_test))
                LRError = np.mean(abs(linearRegressionPrediction - y_test))
                ANNError = np.mean(abs(ANNPrediction - y_test))

                BLErrorCrossValIter.append(BLError)
                LRErrorCrossValIter.append(LRError)
                ANNErrorCrossValIter.append(ANNError)

            BLResult.append(np.array(BLErrorCrossValIter).mean())
            ANNResult.append(np.array(ANNErrorCrossValIter).mean())
            LRResult.append(np.array(LRErrorCrossValIter).mean())

        BLResult = [int(x) for x in BLResult]
        LRResult = [int(x) for x in LRResult]
        ANNResult = [int(x) for x in ANNResult]





        labels = ['5km', '10km', '15km', '20km']

        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        fig, axs = plt.subplots(1,1)

        rects1 = axs.bar(x - width, BLResult, width, label='Baseline')
        rects2 = axs.bar(x, LRResult, width, label='Linear Regression')
        rects3 = axs.bar(x + width, ANNResult, width, label='Neural Network')

        xfmt = md.DateFormatter('%M:%S')



        axs.set_ylabel('Mean absolute error [mm:ss]')
        axs.set_yticks([60,120,180,240,300,360],['1:00','2:00','3:00','4:00','5:00','6:00'])

        #fig.suptitle('Comparison of prediction models')
        axs.set_xticks(x)
        axs.set_xticklabels(labels)

        axs.legend(prop={'size': 12},title='Model')
        axs.set_xlabel('Prediction at split')



        plt.grid(axis='y')
        #fig.tight_layout()
        plt.savefig('figures/MethodComparison.svg',format = 'svg',bbox_inches = "tight")

        #plt.show()

    def AnnWithFeaturesComparison(self):
        plt.clf()
        self.df = self.df.dropna()

        print('Datapoints used in features ANN')
        print(len(self.df))


        self.df['GenderNum'] = self.df['Gender'].map(dict(zip(['M','F'],[-1,1])))

        splits = ['5km', '10km', '15km', '20km']

        usedFeaturesCombinations = [[],['Age','GenderNum'],['Temperature'],['LastYearTime','LastYearSplitDifference'],['Temperature','LastYearTime','LastYearSplitDifference','Age','GenderNum']]
        Results = []





        for usedFeatures in usedFeaturesCombinations:



            FeatureResult = []


            usedColumns = usedFeatures

            for split in splits:

                usedColumns.append(split)

                test = self.df[self.df['Year'] == 2019]
                train = self.df[self.df['Year'] != 2019]





                ANNErrors = []
                for iteration in range(2):
                    X_train = train[usedColumns]
                    X_test = test[usedColumns]

                    y_train = train['Time']
                    y_test = test['Time']

                    y_test = np.array(y_test).reshape(-1, 1)
                    y_train = np.array(y_train).reshape(-1, 1)

                    X_test = np.array(X_test)
                    X_train = np.array(X_train)

                    xScaler = StandardScaler()
                    yScaler = StandardScaler()

                    X_train = xScaler.fit_transform(X_train)
                    X_test = xScaler.transform(X_test)

                    y_train = yScaler.fit_transform(y_train)
                    y_test = yScaler.transform(y_test)
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

                    cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
                    ANNmodel = tf.keras.models.Sequential()
                    ANNmodel.add(tf.keras.layers.Dense(units=40, activation='relu'))
                    ANNmodel.add(tf.keras.layers.Dense(units=1))
                    ANNmodel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics='mae')
                    history = ANNmodel.fit(X_train, y_train, batch_size=1024,
                                           epochs=2000, callbacks=cb, validation_data=(X_val, y_val))

                    ANNPrediction = pd.DataFrame(yScaler.inverse_transform(ANNmodel.predict(X_test)))


                    y_test = yScaler.inverse_transform(y_test).reshape(len(y_test))
                    ANNPrediction = yScaler.inverse_transform(ANNmodel.predict(X_test)).reshape(len(y_test))

                    ANNError = np.mean(abs(ANNPrediction - y_test))

                    ANNErrors.append(ANNError)



                FeatureResult.append(np.array(ANNErrors).mean())



            Results.append(FeatureResult)





        labels = ['5km', '10km', '15km', '20km']

        x = np.arange(len(labels))  # the label locations
        width = 0.15  # the width of the bars

        fig, axs = plt.subplots(1,1)



        rects1 = axs.bar(x - 2*width, Results[0], width, label='Only intermediate splits')
        rects2 = axs.bar(x -width, Results[1], width,label='+ Age & Gender')
        rects3 = axs.bar(x, Results[2], width, label='+ Temperature')
        rects4 = axs.bar(x + width, Results[3], width, label='+ Last race data')
        rects5 = axs.bar(x + 2*width, Results[4], width, label='+ All above')

        def minSecString(val):
            if val < 60:
                return str(val)
            elif val%60 > 9:
                return str(val//60)+':'+str(val%60)
            else:
                return str(val // 60) + ':0' + str(val % 60)



        axs.set_ylabel('Mean absolute error [mm:ss]')
        axs.set_yticks([60,120,180,240,300],['1:00','2:00','3:00','4:00','5:00'])

        axs.set_ylabel('Mean absolute error [mm:ss]')


        axs.set_xticks(x)
        axs.set_xticklabels(labels)


        axs.legend(prop={'size': 12}, title='Features')
        axs.set_xlabel('Prediction at split')

        plt.grid(axis='y')
        #fig.tight_layout()
        plt.savefig('figures/FeatureComparison.svg', format='svg',bbox_inches = "tight")

        #plt.show()

if __name__ == "__main__":
    PacingProject = PacingProject()

    PacingProject.ReadCsv()
    PacingProject.RemoveFaultyData()


    PacingProject.PredictMethodComparison()
    PacingProject.AnnWithFeaturesComparison()




