import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import colorbar, legend
from matplotlib.ticker import PercentFormatter
import os
from os.path import exists

import itertools

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
        self.relativePaces = ['5KmRelativePace', '10KmRelativePace', '15KmRelativePace', '20KmRelativePace',
                         '21KmRelativePace']
        self.boxLimits = [0.98,1.02]
        self.boxLimits.append(10)

    def MakeCSVs(self):
        if(not exists("Varvetresultat/AllResult.csv")):
            #read and merge all results csvs
            for file in self.files:
                temp = pd.read_csv(self.directory+'/'+file, header=0, sep=";")
                #drop not used cols
                temp.drop(['ResultId','RaceId','CountryIso','County','Municipality','ActualStartTime','Place','PlaceClass','PlaceTotal'], axis=1, inplace=True)
                #drop not a number
                temp.dropna(axis=0, how='any', inplace=True)
                #temp = temp.dropna()
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
        self.AddPaces()

    def ReadSmallTestCsv(self): #read small file
        self.df = pd.read_csv(self.directory + '/SmallTestResult.csv')
        self.AddPaces()
        self.AddPB()


    def AddPaces(self):
        self.df['5kmPace'] = (self.df['5km']/5)
        self.df['10kmPace'] = ((self.df['10km'] - self.df['5km']) / 5)
        self.df['15kmPace'] = ((self.df['15km'] - self.df['10km']) / 5)
        self.df['20kmPace'] = ((self.df['20km'] - self.df['15km']) / 5)
        self.df['21kmPace'] = ((self.df['Time'] - self.df['20km']) / 1.0975)
        self.df['TotalPace'] = (self.df['Time']/21.0975)

        self.df['5KmRelativePace'] = self.df['5kmPace']/self.df['TotalPace']
        self.df['10KmRelativePace'] = self.df['10kmPace']/self.df['TotalPace']
        self.df['15KmRelativePace'] = self.df['15kmPace']/self.df['TotalPace']
        self.df['20KmRelativePace'] = self.df['20kmPace']/self.df['TotalPace']
        self.df['21KmRelativePace'] = self.df['21kmPace']/self.df['TotalPace']
        self.BP()
        self.DoS()

    def AddPB(self):
        min_values = self.df.groupby('AthleteId').agg({'Time' : np.min})
        self.df = self.df.merge(min_values, how='outer', on='AthleteId')
        self.df.rename(columns = {'Time_x' : 'Time', 'Time_y':'Pb'}, inplace = True)

    def PrintStat(self,groupBy):
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
        df["SlowdownThresholds"] = pd.Series(np.arange(0.1,0.60, 0.05))

        df["5Km"] = 0
        df["10Km"] = 0
        df["15Km"] = 0
        df["20Km"] = 0
        df["21Km"] = 0

        for i, value in enumerate(df["SlowdownThresholds"]):
            self.LoS(value)
            for j, dist in enumerate([5,10,15,20,21.0975]):
                df.iat[i,j+1] = self.df.loc[(self.df["LoS"] >= dist), ["AthleteId"]].count() / self.df["AthleteId"].count()

        x=df["SlowdownThresholds"]
        plt.plot(x, df["5Km"], 'go--', label="5Km")
        plt.plot(x, df["10Km"],'bD--', label="10Km")
        plt.plot(x, df["15Km"],'yx--', label="15Km")
        plt.legend()
        plt.xlabel("Slowdown Thresholds")
        plt.ylabel("Proportion of runners")
        plt.show()
    
    def PlotSlowdownGroups(self):
        los = 0.25
        self.LoS(los)
        slows = self.df.loc[self.df["LoS"] >= 5, "Time"]

        stats = slows.describe()
        print(stats)
        print("Lower 2.5%: " + str(slows.quantile(q=0.1)))
        print("Upper 97.5%: " + str(slows.quantile(q=0.9)))
        _, ax1 = plt.subplots()
        #weights=np.ones(len(slows)) / len(slows)
        ax1.hist(slows, bins=50, density=True)
        ax1.set_title(f"Groupings of slowdowns for DoS of {los}")
        ax1.set_xlabel("Running time")
        ax1.set_ylabel("Percentage of runners")
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
        slows = self.df.loc[self.df["LoS"] >= 5, "Time"]

        stats = slows.describe()
        print(stats)
        print("Lower 2.5%: " + str(slows.quantile(q=0.025)))
        print("Upper 97.5%: " + str(slows.quantile(q=0.975)))
        _, ax1 = plt.subplots()
        #_, ax2 = plt.subplots()
        #_, ax3 = plt.subplots()
        #_, ax4 = plt.subplots()
        speedupfail = self.df.loc[self.df["LoS"] >= 5, "15kmPace"] / self.df.loc[self.df["LoS"] >= 5, "10kmPace"] 
        speedupSucc = self.df.loc[self.df["LoS"] < 5, "15kmPace"] / self.df.loc[self.df["LoS"] < 5, "10kmPace"]
        ax1.set_xlim([4000, 12000])
        #ax1.set_ylim([0.8, 1.5])
        ax1.scatter(x=self.df.loc[self.df["LoS"] < 5, "Time"], y=speedupSucc, c='b', s=0.2)
        ax1.scatter(x=self.df.loc[self.df["LoS"] >= 5, "Time"], y=speedupfail, c='r', s=0.2)
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

        pd.set_option("display.max_rows", None, "display.max_columns", None,'display.expand_frame_repr', False)
        #print(str((self.df['Time'] == 0).sum())+' runners with 00:00:00 as finish time')
        #print(self.df[self.df['Time'] == 0])
        self.df = self.df[self.df['Time'] != 0]

        self.df = self.df[self.df["AthleteId"] != 0]
        #print('std of last km relative pace')
        #print(np.std(self.df['21KmRelativePace']))


        #print(str((self.df['21KmRelativePace'] > 3).sum())+' runners with 3xslowdown 20-21km')
        #print(self.df[self.df['21KmRelativePace'] > 3])
        self.df = self.df[self.df['21KmRelativePace'] < 3]

        #print(str((self.df['20KmRelativePace'] > 3).sum())+' runners with 3xslowdown 15-20km')
        #print(self.df[self.df['20KmRelativePace'] > 3])

        #print(str((self.df['15KmRelativePace'] > 3).sum())+' runners with 3xslowdown 10-15km')
        #print(self.df[self.df['15KmRelativePace'] > 3])

        #print(str((self.df['10KmRelativePace'] > 3).sum())+' runners with 3xslowdown 5-10km')
        #print(self.df[self.df['10KmRelativePace'] > 3])

        #print(str((self.df['5KmRelativePace'] > 3).sum())+' runners with 3xslowdown 0-5km')
        #print(self.df[self.df['5KmRelativePace'] > 3])




    def ShowDistributions(self):

        #All
        fig, axs = plt.subplots(5, 1)
        for ax,relativePace in zip(axs,self.relativePaces):
            ax.hist(self.df[relativePace],bins=200,range=[0.75,1.25])
        plt.tight_layout()

        #Split on gender
        fig, axs = plt.subplots(5, 1)
        masks = [(self.df['Gender'] == 'M'),(self.df['Gender'] == 'F')]
        colors = ['r','b']
        for ax,relativePace in zip(axs,self.relativePaces):

            for color,mask in zip(colors,masks):
                ax.hist(self.df[mask][relativePace],bins=200,range=[0.75,1.25],histtype='step',color=color,density=1)
        plt.tight_layout()

        #Split on age
        fig, axs = plt.subplots(5, 1)
        masks = [(self.df['Year'] - self.df['Birthyear'] > 20) & (self.df['Year'] - self.df['Birthyear'] > 29),(self.df['Year'] - self.df['Birthyear'] > 30) & (self.df['Year'] - self.df['Birthyear'] > 39),(self.df['Year'] - self.df['Birthyear'] > 40) & (self.df['Year'] - self.df['Birthyear'] > 49),(self.df['Year'] - self.df['Birthyear'] > 50) & (self.df['Year'] - self.df['Birthyear'] > 100)]
        colors = [(0.0, 0.0, 0.0),(0.5, 0.0, 0.0),(1.0, 0.0, 0.0),(1.0, 0.5, 0.5)]
        for ax,relativePace in zip(axs,self.relativePaces):

            for color,mask in zip(colors,masks):
                ax.hist(self.df[mask][relativePace],bins=200,range=[0.75,1.25],histtype='step',color=color,density=1)
        plt.tight_layout()


        #Split on racetime
        fig, axs = plt.subplots(5, 1)
        masks = [(self.df['Time'] < 5400),(self.df['Time'] > 5401) & (self.df['Time'] < 7200),(self.df['Time'] > 7201) & (self.df['Time'] < 9000),(self.df['Time'] > 9000)]
        colors = [(0.0, 0.0, 0.0),(0.5, 0.0, 0.0),(1.0, 0.0, 0.0),(1.0, 0.5, 0.5)]
        for ax,relativePace in zip(axs,self.relativePaces):

            for color,mask in zip(colors,masks):
                ax.hist(self.df[mask][relativePace],bins=200,range=[0.75,1.25],histtype='step',color=color,density=1)
        plt.tight_layout()


        #hot years 2010,2013
        #Split on racetime
        fig, axs = plt.subplots(5, 1)
        masks = [(self.df['Year'] == 2010) | (self.df['Year'] == 2013),(self.df['Year'] != 2010) & (self.df['Year'] != 2013)]
        colors = ['r','b']
        for ax,relativePace in zip(axs,self.relativePaces):

            for color,mask in zip(colors,masks):
                ax.hist(self.df[mask][relativePace],bins=200,range=[0.75,1.25],histtype='step',color=color,density=1)
        plt.tight_layout()
        plt.show()


    def PaceClassification(self):

        '''
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
        plt.show()
        '''


        combinations = list(itertools.product(range(len(self.boxLimits)), repeat=len(self.relativePaces)))

        pacingStrategyDict = {}
        cumSum = 0

        for combination in combinations:
            temp = self.df
            keyString = ''

            for (itemIndex,item) in enumerate(combination):
                previousLimit = 0


                for (boxIndex,limit) in enumerate(self.boxLimits):

                    if item == boxIndex:
                        if boxIndex == 0:
                            keyString += '<'+ str(limit)
                        elif boxIndex == len(self.boxLimits)-1:
                            keyString += '>'+ str(previousLimit)
                        else:
                            keyString += str(previousLimit)+'-'+str(limit)

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

    def HealthCharts(self):
        years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
        means = []
        std = []
        for year in years:
            plt.subplot(10, 1, year - 2009)
            plt.hist(self.df.loc[self.df["Year"] == year, "Time"], bins=50)
            m = self.df.loc[self.df["Year"] == year, "Time"].mean()
            print(m)
            s = self.df.loc[self.df["Year"] == year, "Time"].std()
            print(s)
            means.append(m)
            std.append(s)
            plt.title(str(year))
        plt.tight_layout()
        _, ax1 = plt.subplots()
        ax1.plot(years,means)
        for i,mean in enumerate(means):
            means[i] = mean + std[i]
            ax1.plot(years,std)
        for i,mean in enumerate(means):
            means[i] = mean - std[i]
            ax1.plot(years,std)
        plt.show()

if __name__ == "__main__":
    PacingProject = PacingProject()
    PacingProject.MakeCSVs()       #Run too make a csv of all races in directory with renamed columns and a smaller with all runners that have completed all races
    
    # Automatically adds paces, BasePace and DoS. 
    # If ReadSmallTestCsv then also add Personal Best
    PacingProject.ReadLargeCsv()
    PacingProject.RemoveFaultyData()
    #PacingProject.ReadLargeCsv()
    
    # Plot DoS sensitivity, proportion of runners to slowdown
    #PacingProject.SensitivityPlot()

    #PacingProject.PlotSlowdownGroups()
    PacingProject.Regressions()

    PacingProject.PBtoProportion()

    PacingProject.HealthCharts()

    #PacingProject.ShowDistributions()
    #PacingProject.PaceClassification()
