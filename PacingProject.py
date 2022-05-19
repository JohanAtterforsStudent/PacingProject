import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar, legend
from matplotlib.ticker import PercentFormatter
import os
from os.path import exists
#import tensorflow as tf
#from sklearn.model_selection import train_test_split
#from tensorflow import keras
#from sklearn.linear_model import LinearRegression

import itertools

#from sklearn.neighbors import NearestNeighbors
#from sklearn.cluster import KMeans
#from sklearn.cluster import DBSCAN
#from sklearn.cluster import OPTICS


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

    def ReadLargeCsv(self):  # read large file
        self.df = pd.read_csv(self.directory + '/AllResult.csv')
        self.AddPaces()

    def ReadSmallTestCsv(self):  # read small file
        self.df = pd.read_csv(self.directory + '/SmallTestResult.csv')
        self.AddPaces()
        self.AddPB()

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
        self.BP()
        self.DoS()

    def AddPB(self):
        min_values = self.df.groupby('AthleteId').agg({'Time': np.min})
        self.df = self.df.merge(min_values, how='outer', on='AthleteId')
        self.df.rename(columns={'Time_x': 'Time', 'Time_y': 'Pb'}, inplace=True)

    def PrintStat(self, groupBy):
        pass

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

    def SensitivityPlot(self):
        df = pd.DataFrame()
        df["SlowdownThresholds"] = pd.Series(np.arange(0.05, 0.60, 0.05))

        df["5Km"] = 0
        df["10Km"] = 0
        df["15Km"] = 0
        df["20Km"] = 0
        df["21Km"] = 0

        self.LoS(0.25)
        for i, value in enumerate(df["SlowdownThresholds"]):
            self.LoS(value)
            for j, dist in enumerate([5,10,15,20,21.0975]):
                df.iat[i,j+1] = self.df.loc[(self.df["LoS"] >= dist), ["AthleteId"]].count() / self.df["AthleteId"].count()

        print(self.df.loc[(self.df["LoS"] >= 5), ["AthleteId"]].count() / self.df["AthleteId"].count())
        x = df["SlowdownThresholds"]
        plt.plot(x, df["5Km"], 'go--', label="5Km")
        plt.plot(x, df["10Km"], 'bD--', label="10Km")
        plt.plot(x, df["15Km"], 'yx--', label="15Km")
        plt.annotate(
        '8.59% of runners slowdown >= 25% \nfor 5 km or more.',
        xy=(0.25, 0.085783), xycoords='data',
        xytext=(20, 50), textcoords='offset points',
        arrowprops=dict(arrowstyle="->"))
        plt.legend()
        # plt.plot(x, df["21Km"], color='r')
        plt.xlabel("Slowdown Thresholds [%]")
        plt.ylabel("Proportion of runners [%]")
        plt.savefig('Sensitivity.png', format='png')
        plt.show()
    
    def PlotSlowdownGroups(self):
        self.df = self.df[self.df['Time'] < 18000]
        los = 0.25
        self.LoS(los)
        slows = self.df.loc[self.df["LoS"] >= 5, "Pb"]
        reg = self.df['Pb']

        _, ax1 = plt.subplots()
        sbins, edges = np.histogram(slows, bins=40)
        regbins, _ = np.histogram(reg, bins=edges)
        #discard = range(19,40)
        #sbins = np.delete(sbins, discard)
        #regbins = np.delete(regbins, discard)
        #edges = np.delete(edges, discard)

        fin = []
        for sbin, rbin in zip(sbins, regbins):
            fin.append(sbin / rbin)
            print(sbin, rbin)
        #print(np.diff(edges))660.52
        ax1.bar(edges[:-1], fin, width=np.diff(edges), edgecolor="black", align="edge")

        ax1.set_title(f"Relative number of runners who HTW to their finishing time, DoS of {los}")
        ax1.set_xlabel("Personal Best [s]")
        ax1.set_ylabel("HTW Runners / Total Runners in Pb-group")
        plt.savefig('HTWRunnersToRuntime.png', format='png')
        #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        #_, ax2 = plt.subplots()
        #dnfs = self.df[self.df['Status'] == 'DNF']["Year"]
        #ax2.hist(dnfs, bins=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
        #ax2.set_title("Dnfs each year")
        #ax2.set_xlabel("Year")
        #ax2.set_ylabel("Number of runners")
        plt.show()
    
    def Regressions(self):
        los = 0.25
        self.LoS(los)
        #slows = self.df.loc[self.df["LoS"] >= 5, "Time"]

        #stats = slows.describe()
        #print(stats)
        #print("Lower 2.5%: " + str(slows.quantile(q=0.025)))
        #print("Upper 97.5%: " + str(slows.quantile(q=0.975)))
        _, ax1 = plt.subplots()
        #_, ax2 = plt.subplots()
        #_, ax3 = plt.subplots()
        #_, ax4 = plt.subplots()
        samp = self.df.sample(n=10000, random_state=1)
        samp['RelativeSpeed'] = 0
        samp['RelativeSpeed'] = samp['20kmPace']/samp['BP']
        speedupsucc = samp.loc[samp["LoS"] < 5, "RelativeSpeed"]
        speedupfail = samp.loc[samp["LoS"] >= 5, "RelativeSpeed"]
        #samp.loc[samp["LoS"] < 5, "RelativeSpeed"] = self.df['10kmPace']/self.df['5kmPace']
        
        #speedupfail = samp.loc[samp["LoS"] >= 5, "21kmPace"] / samp.loc[samp["LoS"] >= 5, "20kmPace"] 
        #speedupSucc = samp.loc[samp["LoS"] < 5, "21kmPace"] / samp.loc[samp["LoS"] < 5, "20kmPace"]
        ax1.set_xlim([1000, 15000])
        ax1.set_ylim([0.8, 1.5])
        ax1.scatter(x=samp.loc[samp["LoS"] < 5, "Time"], y=speedupsucc, c='b', s=0.2)
        ax1.scatter(x=samp.loc[samp["LoS"] >= 5, "Time"], y=speedupfail, c='r', s=0.2)
        ax1.set_title(f"Speedups between 15 Km and Base Pace at DoS of {los}")
        ax1.set_xlabel("Running time [Sec]")
        ax1.set_ylabel("Relative pace speedup [1]")
        #plt.savefig('SpeedupBP-15.png', format='png')
        #ax2.set_xlim([4000, 11000])
        #ax2.set_ylim([0.8, 1.5])
        #ax2.scatter(x=slows, y=self.df.loc[self.df["LoS"] >= 5, "10kmPace"])
        #ax3.scatter(x=slows, y=self.df.loc[self.df["LoS"] >= 5, "15kmPace"])
        #ax4.scatter(x=slows, y=self.df.loc[self.df["LoS"] >= 5, "20kmPace"])
        plt.show()
        
    def PBtoProportion(self):
        if "Pb" not in self.df:
            print("AllResults.csv does not calculate a Pb, try with SmallTestResults.csv")
            return
        los = 0.3
        self.LoS(los)
        slows = self.df.loc[self.df["LoS"] >= 5]
        prop = slows["Time"] / slows["Pb"]
        _, ax1 = plt.subplots()
        ax1.hist(prop.unique(), bins=50, density=True)
        print("Statistics for proportionality of slowdown to Pb")
        print(prop.describe())
        ax1.set_title(f"Proportion of finish time to personal best at DoS {los}")
        ax1.set_ylabel("Number of runners")
        ax1.set_xlabel("Ratio Time / Pb")

        _, ax2 = plt.subplots()
        ax2.hist(self.df["Pb"], bins=50, density=True)
        print("------------------------------------------------")
        print("Statistics for personal best")
        print(self.df["Pb"].describe())
        ax2.set_title("Personal best distribution")
        ax2.set_ylabel("Number of runners")
        ax2.set_xlabel("Personal best")
        plt.show()


    def RemoveFaultyData(self):

        print('Removing data containg errors')

        pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.expand_frame_repr', False)
        # print(str((self.df['Time'] == 0).sum())+' runners with 00:00:00 as finish time')
        # print(self.df[self.df['Time'] == 0])
        self.df = self.df[self.df['Time'] != 0]

        # print('std of last km relative pace')
        # print(np.std(self.df['21KmRelativePace']))

        # print(str((self.df['21KmRelativePace'] > 3).sum())+' runners with 3xslowdown 20-21km')
        # print(self.df[self.df['21KmRelativePace'] > 3])
        self.df = self.df[self.df['21KmRelativePace'] < 2]
        self.df = self.df[self.df['20KmRelativePace'] < 2]
        self.df = self.df[self.df['15KmRelativePace'] < 2]
        self.df = self.df[self.df['10KmRelativePace'] < 2]
        self.df = self.df[self.df['5KmRelativePace'] < 2]
        self.df = self.df[self.df["AthleteId"] != 0]
        #print('std of last km relative pace')
        #print(np.std(self.df['21KmRelativePace']))


        # print(str((self.df['20KmRelativePace'] > 3).sum())+' runners with 3xslowdown 15-20km')

        # print(str((self.df['15KmRelativePace'] > 3).sum())+' runners with 3xslowdown 10-15km')
        # print(self.df[self.df['15KmRelativePace'] > 3])

        # print(str((self.df['10KmRelativePace'] > 3).sum())+' runners with 3xslowdown 5-10km')
        # print(self.df[self.df['10KmRelativePace'] > 3])

        # print(str((self.df['5KmRelativePace'] > 3).sum())+' runners with 3xslowdown 0-5km')
        # print(self.df[self.df['5KmRelativePace'] > 3])

    def ShowDistributions(self):
        plt.rc('font', size=5)
        # All
        names = ['0-5km', '5-10km', '10-15km', '15-20km', '20-21,1km']
        fig, axs = plt.subplots(5, 1)
        fig.suptitle('Distribution of pace for different segments all runners')
        for ax, relativePace, name in zip(axs, self.relativePaces, names):
            ax.hist(self.df[relativePace], bins=200, range=[0.75, 1.25], density=1, histtype='step')
            ax.set_xlabel('Relative pace')
            ax.set_ylabel('Density')
            ax.title.set_text(name)

        plt.tight_layout()
        plt.savefig('all.png', format='png')

        # Split on gender
        fig, axs = plt.subplots(5, 1)
        fig.suptitle('Distribution of pace for different segments split on gender')
        masks = [(self.df['Gender'] == 'M'), (self.df['Gender'] == 'F')]
        colors = ['r', 'b']
        labels = ['F', 'M']
        for ax, relativePace, name in zip(axs, self.relativePaces, names):
            for color, mask, label in zip(colors, masks, labels):
                ax.hist(self.df[mask][relativePace], bins=200, range=[0.75, 1.25], histtype='step', color=color,
                        density=1)
            ax.title.set_text(name)
            ax.legend(labels)
            ax.set_xlabel('Relative pace')
            ax.set_ylabel('Density')
        plt.tight_layout()
        plt.savefig('gender.png', format='png')

        # Split on age
        fig, axs = plt.subplots(5, 1)
        fig.suptitle('Distribution of pace for different segments split on age')
        masks = [(self.df['Year'] - self.df['Birthyear'] > 25),
                 (self.df['Year'] - self.df['Birthyear'] > 26) & (self.df['Year'] - self.df['Birthyear'] > 36),
                 (self.df['Year'] - self.df['Birthyear'] > 35)]
        colors = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
        labels = ['<26', '26-35', '>36']
        for ax, relativePace, name in zip(axs, self.relativePaces, names):

            for color, mask in zip(colors, masks):
                ax.hist(self.df[mask][relativePace], bins=200, range=[0.75, 1.25], histtype='step', color=color,
                        density=1)
            ax.title.set_text(name)
            ax.legend(labels)
            ax.set_xlabel('Relative pace')
            ax.set_ylabel('Density')
        plt.tight_layout()
        plt.savefig('age.png', format='png')

        # Split on racetime
        fig, axs = plt.subplots(5, 1)
        fig.suptitle('Distribution of pace for different segments split on finishing time')
        masks = [(self.df['Time'] < 5400), (self.df['Time'] > 5401) & (self.df['Time'] < 7200),
                 (self.df['Time'] > 7201) & (self.df['Time'] < 9000), (self.df['Time'] > 9000)]
        colors = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.5, 0.5)]
        labels = ['<1:30', '1:30-2:00', '2:00-2:30', '>2:30']
        for ax, relativePace, name in zip(axs, self.relativePaces, names):

            for color, mask in zip(colors, masks):
                ax.hist(self.df[mask][relativePace], bins=200, range=[0.75, 1.25], histtype='step', color=color,
                        density=1)
            ax.title.set_text(name)
            ax.legend(labels)
            ax.set_xlabel('Relative pace')
            ax.set_ylabel('Density')
        plt.tight_layout()
        plt.savefig('time.png', format='png')

        # hot years 2010,2013
        # Split on racetime
        fig, axs = plt.subplots(5, 1)
        fig.suptitle('Distribution of pace for different segments split on temperature')
        masks = [(self.df['Year'] == 2010) | (self.df['Year'] == 2013),
                 (self.df['Year'] != 2010) & (self.df['Year'] != 2013)]
        colors = ['r', 'b']
        labels = ['Hot', 'Cold']
        for ax, relativePace, name in zip(axs, self.relativePaces, names):

            for color, mask in zip(colors, masks):
                ax.hist(self.df[mask][relativePace], bins=200, range=[0.75, 1.25], histtype='step', color=color,
                        density=1)
            ax.title.set_text(name)
            ax.legend(labels)
            ax.set_xlabel('Relative pace')
            ax.set_ylabel('Density')
        plt.tight_layout()
        plt.savefig('temperature.png', format='png')
        plt.show()

    def PaceClassification(self):

        X = self.df[self.relativePaces]
        kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
        print('KMeans clusters')
        print(kmeans.cluster_centers_)

        distorsions = []
        for k in range(2, 10):
            print(k)
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            distorsions.append(kmeans.inertia_)

        plt.plot(range(2, 10), distorsions)
        plt.title('Elbow curve')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')

        plt.savefig('elbow.png', format='png')
        plt.show()

        combinations = list(itertools.product(range(len(self.boxLimits)), repeat=len(self.relativePaces)))

        pacingStrategyDict = {}
        cumSum = 0

        for combination in combinations:
            temp = self.df
            keyString = ''

            for (itemIndex, item) in enumerate(combination):
                previousLimit = 0

                for (boxIndex, limit) in enumerate(self.boxLimits):

                    if item == boxIndex:
                        if boxIndex == 0:
                            keyString += '<' + str(limit)
                        elif boxIndex == len(self.boxLimits) - 1:
                            keyString += '>' + str(previousLimit)
                        else:
                            keyString += str(previousLimit) + '-' + str(limit)

                        temp = temp[temp[self.relativePaces[itemIndex]] > previousLimit]
                        temp = temp[temp[self.relativePaces[itemIndex]] <= limit]

                    previousLimit = limit
                keyString += '  '

            pacingStrategyDict[keyString] = len(temp)

        print(sorted(pacingStrategyDict, key=pacingStrategyDict.get, reverse=True))

        '''
        db = DBSCAN(eps=0.05, min_samples=2*5).fit(X)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        '''

    def ANNPredict(self):

        BLResult = []
        LRResult = []
        ANNResult = []

        splits = ['5km', '10km', '15km', '20km']
        passedSplits = []

        self.LoS(0.25)
        self.df.loc[self.df["LoS"] >= 5, 'HTW'] = 1
        self.df.loc[self.df["LoS"] < 5, 'HTW'] = 0

        for split in splits:

            passedSplits.append(split)
            y = self.df['Time']
            X = self.df[passedSplits]

            y = np.array(y)
            X = np.array(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            # ANN model
            cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
            ANNmodel = tf.keras.models.Sequential()
            ANNmodel.add(tf.keras.layers.Dense(units=4, activation='relu'))
            ANNmodel.add(tf.keras.layers.Dense(units=1))
            ANNmodel.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics='mae')
            history = ANNmodel.fit(X_train, y_train, batch_size=1024,
                                   epochs=500, callbacks=cb, validation_data=(X_test, y_test))

            ANNPrediction = pd.DataFrame(ANNmodel.predict(X_test)).values.reshape(-1)

            # baseline
            nPassedSplits = len(passedSplits)
            if nPassedSplits == 1:
                lastPace = X_test[:, 0] / 5
            else:
                lastPace = (X_test[:, nPassedSplits - 1] - X_test[:, nPassedSplits - 2]) / 5

            baselinePrediction = X_test[:, nPassedSplits - 1] + lastPace * (1.0975 + (4 - nPassedSplits) * 5)

            # LR
            LRmodel = LinearRegression()
            LRmodel.fit(X_train, y_train)
            linearRegressionPrediction = LRmodel.predict(X_test)

            ANNError = (1 / (len(ANNPrediction)) * sum((ANNPrediction - y_test) ** 2)) ** 0.5
            BLError = (1 / (len(baselinePrediction)) * sum((baselinePrediction - y_test) ** 2)) ** 0.5
            LRError = (1 / (len(linearRegressionPrediction)) * sum((linearRegressionPrediction - y_test) ** 2)) ** 0.5

            BLResult.append(BLError)
            ANNResult.append(ANNError)
            LRResult.append(LRError)

        print(BLResult)
        print(ANNResult)
        print(LRResult)

        '''
        fig, ax = plt.subplots()
        ax.hist(baselineError,bins=120,range=[0,1200],histtype='step',color='b',density=1)
        ax.hist(predictionError, bins=120, range=[0, 1200], histtype='step', color='r', density=1)
        plt.tight_layout()
        plt.show()
        '''


if __name__ == "__main__":
    PacingProject = PacingProject()
    PacingProject.MakeCSVs()  # Run too make a csv of all races in directory with renamed columns and a smaller with all runners that have completed all races

    # Automatically adds paces, BasePace and DoS. 
    # If ReadSmall then also add Personal Best
    #PacingProject.ReadSmallTestCsv()
    #PacingProject.ReadLargeCsv()
    #PacingProject.RemoveFaultyData()

    # Plot DoS sensitivity, proportion of runners to slowdown
    #PacingProject.SensitivityPlot()

    #PacingProject.PlotSlowdownGroups()

    #PacingProject.Regressions()

    #PacingProject.PBtoProportion()

    #PacingProject.HealthCharts()

    #PacingProject.ShowDistributions()
    #PacingProject.PaceClassification()

    #PacingProject.ANNPredict()
