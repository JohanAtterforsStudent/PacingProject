import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import Base 

class htw:
    def __init__(self):
        file = ""
        self.df = pd.read_csv(file, header=0, sep=";")

    def SensitivityPlot(self):
        df = pd.DataFrame()
        df["SlowdownThresholds"] = pd.Series(np.arange(0.05, 0.60, 0.05))

        df["5Km"] = 0
        #df["10Km"] = 0
        #df["11Km"] = 0

        for i, value in enumerate(df["SlowdownThresholds"]):
            self.LoS(value)
            for j, dist in enumerate([1]):
                df.iat[i,j+1] = self.df.loc[(self.df["LoS"] >= 5), ["AthleteId"]].count() / self.df["AthleteId"].count()

        self.LoS(0.25)

        print(self.df.loc[(self.df["LoS"] >= 5), ["AthleteId"]].count() / self.df["AthleteId"].count())
        print(self.df["AthleteId"].count())
        print(self.df.loc[(self.df["LoS"] >= 5), ["AthleteId"]].count())
    
        x = df["SlowdownThresholds"]
        print(df['5Km'])
        print(df["SlowdownThresholds"])
        plt.plot(x, df["5Km"], 'o--b', label="5Km")
        #plt.plot(x, df["10Km"], 'D--r', label="10Km")
        #plt.plot(x, df["11Km"], 'x--b', label="11Km")
        plt.annotate(
        '8.6% of runners \nslowdown >= 25% \nfor 5 km or more.',
        xy=(0.25, 0.086), xycoords='data',
        xytext=(-15, 80), textcoords='offset points',
        arrowprops=dict(arrowstyle="->"))

        plt.legend()
        #plt.plot(x, df["21Km"], color='r')
        plt.xlabel("Slowdown Thresholds")
        plt.ylabel("Proportion of runners")
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/Sensitivity.pdf')
        plt.show()

    def GetXy(self):
        dist = 10
        # Preprocess
        self.LoS(0.25)
        df = pd.DataFrame()
        df = self.df[['5kmPace', '10kmPace', '15kmPace', '20kmPace',
           '21kmPace', 'LoS', 'temperature', 'LastTime', 'LastPaceGroup', 'Gender', 'LastSplitRatio', 'Age', 'Runs']].copy()

        df = pd.get_dummies(df, columns= ['Gender'], drop_first=True)
        df = pd.get_dummies(df, columns= ['LastPaceGroup'])

        # Labeling 
        df['HTW'] = 0
        df.loc[df['LoS'] >= 5, 'HTW'] = 1
        df.drop(['LoS'], axis=1, inplace=True)

        if dist == 0:
          df.drop(['5kmPace', '10kmPace', '15kmPace', '20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 5:
          df.drop(['10kmPace', '15kmPace', '20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 10:
          df.drop(['15kmPace', '20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 15:
          df.drop(['20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 20:
          df.drop(['21kmPace'], axis=1, inplace=True)

        delcols = int(5 - (dist / 5))

        temp = df.drop(['LastPaceGroup_EVEN', 'LastPaceGroup_HTW', 'LastPaceGroup_NEG', 'LastPaceGroup_POS'], axis=1, inplace=False)
        #print(temp.head())
        #print(temp['Runs'].value_counts())
        tempdata = df.values
        Xtemp = tempdata[:, 0:8].astype('float32')
        #print(len(Xtemp[0, 0:8]))

        data = df.values
        X = data[:,0:12 - delcols].astype('float32')
        y = data[:,-1].astype('float32')
        return (Xtemp, y)

    def RForest(self):
        dist = 10
        # Preprocess
        self.LoS(0.25)
        df = pd.DataFrame()
        df = self.df[['5kmPace', '10kmPace', '15kmPace', '20kmPace',
           '21kmPace', 'LoS', 'temperature', 'LastTime', 'LastPaceGroup', 'Gender', 'LastSplitRatio', 'Age', 'Runs']].copy()

        df = pd.get_dummies(df, columns= ['Gender'], drop_first=True)
        df = pd.get_dummies(df, columns= ['LastPaceGroup'])

        # Labeling 
        df['HTW'] = 0
        df.loc[df['LoS'] >= 5, 'HTW'] = 1
        df.drop(['LoS'], axis=1, inplace=True)

        if dist == 0:
          df.drop(['5kmPace', '10kmPace', '15kmPace', '20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 5:
          df.drop(['10kmPace', '15kmPace', '20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 10:
          df.drop(['15kmPace', '20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 15:
          df.drop(['20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 20:
          df.drop(['21kmPace'], axis=1, inplace=True)

        delcols = int(5 - (dist / 5))

        temp = df.drop(['LastPaceGroup_EVEN', 'LastPaceGroup_HTW', 'LastPaceGroup_NEG', 'LastPaceGroup_POS'], axis=1, inplace=False)
        print(temp.head())
        print(temp['Runs'].value_counts())
        tempdata = df.values
        Xtemp = tempdata[:, 0:8].astype('float32')
        print(len(Xtemp[0, 0:8]))


        data = df.values
        X = data[:,0:12 - delcols].astype('float32')
        y = data[:,-1].astype('float32')

        #compute_sample_weight('balanced', np.unique(self.y), self.y)
        X_train, X_test, y_train, y_test = train_test_split(Xtemp, y, test_size=0.1, stratify=y, random_state=1)

        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt', 'log2']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 200, num = 20)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10, 15, 20]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [int(x) for x in np.linspace(1, 20, num = 10)]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]# Create the random grid
        criterion = ['gini', 'entropy']
        classweight = [{0:1, 1: w} for w in [int(x) for x in np.linspace(0.5, 100, num = 20)]]

        random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'criterion': criterion}
                   #'class_weight': classweight}

        clf = BalancedRandomForestClassifier() # sampling_strategy= 0.5, class_weight='balanced' n_estimators=150, random_state=0, class_weight={0:1, 1:10}

        #rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, 
        #                               n_iter = 10, cv = 3, verbose=2, random_state=42, scoring='f1', n_jobs=1)

        #rf_random.fit(X_train, y_train)

        #print(rf_random.best_params_)

        print("Fitting the model ...")

        #params1 = {'n_estimators': 1200, 'min_samples_split': 20, 'min_samples_leaf': 11, 'max_features': 'sqrt', 'max_depth': 110, 'criterion': 'gini', 'bootstrap': True}
        #params_age = {'n_estimators': 2000, 'min_samples_split': 10, 'min_samples_leaf': 15, 'max_features': 'log2', 'max_depth': 170, 'criterion': 'entropy', 'bootstrap': False}
        cf.set_params(n_estimators=2000, min_samples_split=10, min_samples_leaf=15, max_features='log2', max_depth=170, criterion='entropy', bootstrap=False)
        cf.fit(X_train, y_train)
        #oversample = SMOTE(sampling_strategy=0.3)
        #ada = ADASYN(random_state=42)
        #over_X, over_y = ada.fit_resample(X_train, y_train)
        #over_X, over_y = oversample.fit_resample(X_train, y_train)
        #over_X_train, over_X_test, over_y_train, over_y_test = train_test_split(over_X, over_y, test_size=0.1, stratify=over_y)
        #Build SMOTE SRF model
        #under = RandomUnderSampler(sampling_strategy=0.8)
        #res_X, res_y = under.fit_resample(over_X, over_y)
        #Create Stratified K-fold cross validation
        #model = EasyEnsembleClassifier(n_estimators=10)
        # define evaluation procedure
        #cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
        #scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=1)
        #print('Mean ROC AUC: %.3f' % mean(scores))
        #print("Fitting the model ...")
        #clf.fit(X_train, y_train)
        '''
        clf = DecisionTreeClassifier()
        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('over', over), ('under', under), ('model', clf)]
        pipeline = Pipeline(steps=steps)
        # evaluate pipeline
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)
        scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
        print('Mean ROC AUC: %.3f' % mean(scores))
        Regular rf
        {'n_estimators': 1200, 'min_samples_split': 2, 'min_samples_leaf': 7, 'max_features': 'sqrt', 'max_depth': 70, 'criterion': 'entropy', 'class_weight': {0: 1, 1: 14}, 'bootstrap': True}
        Balancd rf
        {'n_estimators': 1200, 'min_samples_split': 20, 'min_samples_leaf': 11, 'max_features': 'sqrt', 'max_depth': 110, 'criterion': 'gini', 'bootstrap': True}
        Balanced rf LastPace
        {'n_estimators': 1800, 'min_samples_split': 5, 'min_samples_leaf': 11, 'max_features': 'sqrt', 'max_depth': 110, 'criterion': 'entropy', 'bootstrap': False}
        With age and runs:
        {'n_estimators': 2000, 'min_samples_split': 10, 'min_samples_leaf': 15, 'max_features': 'log2', 'max_depth': 170, 'criterion': 'entropy', 'bootstrap': False}
        '''

        joblib.dump(clf, '/content/drive/MyDrive/Varvetresultat/Models/RandomForestBalancedCV_Age.pkl')
        # clf = joblib.load('/content/drive/MyDrive/Varvetresultat/Models/RandomForest.pkl')
        y_pred = clf.predict(X_test)
        #Create confusion matrix
        fig, ax = plt.subplots()
        #plt.rcParams.update({'font.size': 20})
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Avoid HTW', 'HTW'], ax=ax, cmap='Blues', normalize='all')
        #plt.title('Confusion Matrix')
        #plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/ConfusionMatrixRForest.pdf')
        plt.show()

        print(metrics.classification_report(y_test, y_pred, target_names=['Avoid HTW', 'HTW']))
        print(dict(zip(df.columns, clf.feature_importances_)))
        '''
        print("Calculating Feature importances ...")
        result = permutation_importance(
          clf, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1
        )
        print(temp.columns)
        forest_importances = pd.Series(result.importances_mean, index=(temp.drop(['HTW'], inplace=False, axis=1)).columns) #index=df.drop(['HTW', 'LastPaceGroup_EVEN', 'LastPaceGroup_HTW', 'LastPaceGroup_NEG', 'LastPaceGroup_POS']
        forest_importances.sort_values(ascending=False, inplace=True) # Decending order
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        #ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        ax.set_ylabel("Feature of a runner")
        fig.tight_layout()
        plt.show()
        '''

        samples = len(y_test) - 1
        print(samples)
        ax = plt.gca()
        rfc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, name='Balanced Random Forest')
        plt.show()


        '''
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scale = min_max_scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
        rf = RandomForestClassifier(n_estimators=100, random_state=1, verbose=True)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print(metrics.classification_report(y_test, y_pred, target_names=['Avoid HTW', 'HTW']))
        print(dict(zip(df.columns, rf.feature_importances_)))
        #plt.figure(figsize=(25,10))
        #fig = plot_tree(rf.estimators_[0], feature_names=['5kmPace', '10kmPace', '15kmPace', '20kmPace','21kmPace'],
        #                class_names=['0','1'], filled=True, rounded=True, fontsize=14)
        #plt.savefig("forest.png")
        '''

    def Paceings(self):
        self.LoS(0.25)
        self.df['HTW'] = 0
        self.df.loc[self.df['LoS'] >= 5, 'HTW'] = 1
        names = ['0-5km', '5-10km', '10-15km', '15-20km', '20-21,1km']

        mean_paces_htw = self.df.loc[self.df['HTW'] == 1, ['5KmRelativePace', 
                                                              '10KmRelativePace', '15KmRelativePace', 
                                                              '20KmRelativePace', '21KmRelativePace']].mean()
        std_paces_htw = self.df.loc[self.df['HTW'] == 1, ['5KmRelativePace', 
                                                              '10KmRelativePace', '15KmRelativePace', 
                                                              '20KmRelativePace', '21KmRelativePace']].std()
        mean_paces = self.df[['5KmRelativePace', 
                                '10KmRelativePace', '15KmRelativePace', 
                                '20KmRelativePace', '21KmRelativePace']].mean()
        std_paces = self.df[['5KmRelativePace', 
                                '10KmRelativePace', '15KmRelativePace', 
                                '20KmRelativePace', '21KmRelativePace']].std()
        fig, ax = plt.subplots()
        trans_mean = Affine2D().translate(-0.05, 0.0) + ax.transData
        trans_htw = Affine2D().translate(+0.05, 0.0) + ax.transData
        ax.errorbar(names, mean_paces.values, yerr=std_paces.values, fmt='--o', capsize=4, transform=trans_mean, label='Mean Paces')
        ax.errorbar(names, mean_paces_htw.values, yerr=std_paces_htw.values, fmt='-o', capsize=4, transform=trans_htw, label='HTW Paces')
        #ax.set_yticks(np.arange(0.8, 1.25, 0.5))
        #ax.axes.get_yaxis().set_ticks(np.arange(0.8, 1.25, 0.5))
        plt.legend()
        plt.grid()
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/NewPacesHTW.pdf')
        plt.show()
        #plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/PacesHTW.pdf')

    def Ratios(self):
        self.df.drop_duplicates(inplace=True)
        self.LoS(0.25)
        self.df['HTW'] = 0
        self.df.loc[self.df['LoS'] >= 5, 'HTW'] = 1
        self.df['Run'] = 0
        self.df.sort_values(by=['Year'], inplace=True)
        self.df['Run'] = self.df.groupby(['AthleteId']).cumcount() +1
        #print(self.df.loc[self.df['AthleteId'] == 155257, ['AthleteId', 'Year', 'Run']])

        dewPoints = [16.4,8.6,3.4,18,10.0,6.0,11.3,10.7,10.5,13.6]
        temperatures = [21.7,16.6,13.6,25.0,18.9,14.7,15.1,13.9,20.0,19.4]

        probsRun = []
        for i in range(1,11):
          htw = self.df.loc[((self.df['HTW'] == 1) & (self.df['Run'] == i)), 'AthleteId'].count()
          all = self.df.loc[self.df['Run'] == i, 'AthleteId'].count()
          probsRun.append(round(htw / all, 5))
          #print(round(htw / all, 5))

        #plt.figure()
        #plt.axis([1, 10, 0.04, 0.12])
        #plt.plot(range(1, 11), probsRun, label='HTW')
        #plt.xlabel('Run')
        #plt.ylabel('Ratio of runners hitting the wall')
        #plt.grid()
        #plt.show()

        probsYear = []
        MprobsYear = []
        FprobsYear = []
        for year in range(2010,2020):
          htw = self.df.loc[((self.df['HTW'] == 1) & (self.df['Year'] == year)), 'AthleteId'].count()
          Mhtw = self.df.loc[((self.df['HTW'] == 1) & (self.df['Year'] == year) & (self.df['Gender'] == 'M')), 'AthleteId'].count()
          Fhtw = self.df.loc[((self.df['HTW'] == 1) & (self.df['Year'] == year) & (self.df['Gender'] == 'F')), 'AthleteId'].count()
          all = self.df.loc[self.df['Year'] == year, 'AthleteId'].count()
          Mall = self.df.loc[((self.df['Year'] == year) & (self.df['Gender'] == 'M')), 'AthleteId'].count()
          Fall = self.df.loc[((self.df['Year'] == year) & (self.df['Gender'] == 'F')), 'AthleteId'].count()
          probsYear.append(round(htw / all, 5))
          MprobsYear.append(round(Mhtw / Mall, 5))
          FprobsYear.append(round(Fhtw / Fall, 5))
          #print(round(htw / all, 5))
          #plt.rcParams.update({'font.size': 12})
        fig, ax1 = plt.subplots(figsize=(10,7))
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Ratio of runners hitting the wall')
        ax1.set_ylim([0, 0.2])
        ax1.set_xlim([2010, 2019])
        #ax1.plot(range(2010, 2020), probsYear,'--', label='Ratio All')
        ax1.plot(range(2010, 2020), MprobsYear, '-+', label='Ratio Male')

        ax1.plot(range(2010, 2020), FprobsYear,'-x', label='Ratio Female')
        #ax1.legend()
        ax1.grid()

        ax2 = ax1.twinx()   
        ax2.set_ylabel('Temperature')
        ax2.plot(range(2010, 2020), temperatures, 'm--', label='Temperature')
        ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))

        fig.legend(bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
        #ax2.tick_params(axis ='y')
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/tempToHtw.pdf')
        plt.show()

    def LossTime(self):
        #self.df.drop_duplicates(inplace=True)
        self.LoS(0.25)
        self.df['HTW'] = 0
        self.df.loc[self.df['LoS'] >= 1, 'HTW'] = 1

        #firstSplitsFail = self.df.loc[(self.df['HTW'] == 1), ['5kmPace', '10kmPace', 'Time']]
        df = pd.DataFrame()
        df['Actual'] = self.df.loc[self.df['HTW'] == 1, 'Time']
        df['Base'] = self.df.loc[self.df['HTW'] == 1, 'BP']

        #df['AvgPace'] = (firstSplitsFail['5kmPace'] + firstSplitsFail['10kmPace']) / 2
        #df['PredTime'] = df['AvgPace']*21.0975

        #df['Ratio'] = firstSplitsFail['Time'] / df['PredTime']

        df['Ratio'] = df['Actual'] / (df['Base'] * 21.0975)

        fig, ax1 = plt.subplots()
        hist, edges = np.histogram(df['Ratio'], bins=25)

        ax1.bar(edges[:-1], hist, width=np.diff(edges), edgecolor="black", align="edge")

        #ax1.set_title(f"Relative number of runners who HTW to their personal best")
        ax1.set_xlabel("Relative slowdown to prevoius finish time")
        ax1.set_ylabel("Number of runners")
        #plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/SlowdownDist.pdf')
        plt.show()

        print(len(self.df.index))
        self.df = self.df[self.df["LastTime"] != 0]
        print(len(self.df.index))

        Mhtw = self.df.loc[(self.df['HTW'] == 1) & (self.df['Gender'] == 'M'), ['Time', 'LastTime']]
        Fhtw = self.df.loc[(self.df['HTW'] == 1) & (self.df['Gender'] == 'F'), ['Time', 'LastTime']]

        Mall = self.df.loc[(self.df['Gender'] == 'M'), ['Time', 'LastTime']]
        Fall = self.df.loc[(self.df['Gender'] == 'F'), ['Time', 'LastTime']]

        MhtwEst = (Mhtw['Time']) / Mhtw['LastTime']
        FhtwEst = (Fhtw['Time']) / Fhtw['LastTime']

        MallEst = (Mall['Time']) / Mall['LastTime']
        FallEst = (Fall['Time']) / Fall['LastTime']

        Mallhist, medges = np.histogram(MallEst, bins=10)
        Mhtwhist, _ = np.histogram(MhtwEst, bins=medges)

        Fallhist, fedges = np.histogram(FallEst, bins=10)
        Fhtwhist, _ = np.histogram(FhtwEst, bins=fedges)

        mhist = []
        for mhtwbin, mallmbin in zip(Mhtwhist, Mallhist):
          mhist.append(mhtwbin / mallmbin)

        fhist = []
        for fhtwbin, fallmbin in zip(Fhtwhist, Fallhist):
          fhist.append(fhtwbin / fallmbin)

        fig, ax1 = plt.subplots(figsize=(10,7))
        ax1.set_xlabel('Relative loss in finish time')
        ax1.set_ylabel('Runners hitting the wall')

        ax1.plot(medges[:-1], mhist, color='orange', label='Male')
        ax1.plot(fedges[:-1], fhist, color='blue', label='Female')
        plt.legend()
        ax1.grid()
        plt.show()

        print(df['Ratio'].mean())
        print(df['Ratio'].std())

        self.df['LoS10km'] = 0
        self.df['LoS15km'] = 0
        self.df['LoS20km'] = 0
        self.df['LoS21km'] = 0

        self.df['LoS15_20km'] = 0
        self.df['LoS20km_Time'] = 0

        self.df['LoS15km_20km_Time'] = 0

        dos = 0.25
        #self.df.loc[(self.df['DoS10km'] >= dos), 'LoS10km'] = 1
        self.df.loc[(self.df['DoS15km'] >= dos), 'LoS15km'] = 1
        self.df.loc[(self.df['DoS20km'] >= dos), 'LoS20km'] = 1
        self.df.loc[(self.df['DoSTime'] >= dos), 'LoS21km'] = 1

        self.df.loc[(self.df['DoS15km'] >= dos) & (self.df['DoS20km'] < dos) & (self.df['DoSTime'] < dos), 'LoS15km'] = 1
        self.df.loc[(self.df['DoS20km'] >= dos) & (self.df['DoS15km'] < dos) & (self.df['DoSTime'] < dos), 'LoS20km'] = 1
        self.df.loc[(self.df['DoSTime'] >= dos) & (self.df['DoS15km'] < dos) & (self.df['DoS20km'] < dos), 'LoS21km'] = 1

        self.df.loc[(self.df['DoS15km'] >= dos) & (self.df['DoS20km'] >=dos) & (self.df['DoSTime'] < dos), 'LoS15_20km'] = 1
        self.df.loc[(self.df['DoS20km'] >= dos) & (self.df['DoSTime'] >= dos) & (self.df['DoS15km'] < dos), 'LoS20km_Time'] = 1

        self.df.loc[(self.df['DoS15km'] >= dos) & (self.df['DoS20km'] >= dos) & (self.df['DoSTime'] >= dos), 'LoS15km_20km_Time'] = 1

        los_base = np.array([self.df['LoS15km'].sum(), self.df['LoS20km'].sum(), self.df['LoS21km'].sum()])

        los_2segs = np.array([self.df['LoS15_20km'].sum(), self.df['LoS15_20km'].sum() + self.df['LoS20km_Time'].sum(), self.df['LoS20km_Time'].sum()])

        los_all = np.array([self.df['LoS15km_20km_Time'].sum(), self.df['LoS15km_20km_Time'].sum(), self.df['LoS15km_20km_Time'].sum()])


        fig, ax1 = plt.subplots(figsize=(10,5))
        ax1.set_xlabel('Segment')
        ax1.set_ylabel('Runners hitting the wall')

        segments_los = ['10-15 km', '15-20 km', '20-21 km']#'0-10 km', 
        print(los_base)
        ax1.bar(x=segments_los, height=los_base, color='cornflowerblue', edgecolor='k', linewidth=1)
        ax1.bar(x=segments_los, height=los_2segs, bottom=los_base, color='royalblue', edgecolor='k', linewidth=1)
        ax1.bar(x=segments_los, height=los_all, bottom=los_base+los_2segs, color='blue', edgecolor='k', linewidth=1)
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/HtwSegments.pdf')
        plt.show()
        self.df.loc[(self.df['DoS20km'] >= dos), 'LoS20km'] = 1
        print(self.df['LoS20km'].sum())
        #temp = self.df.drop(["count", '5KmRelativePace', '10KmRelativePace', '15KmRelativePace', '20KmRelativePace', '21KmRelativePace', '10KmRP', '15KmRP', '20KmRP', '21KmRP', 'FirstHalfPace', 'SecondHalfPace', 'SplitRatio', 
        #                     'DoS15km', 'DoS20km', 'DoSTime', 'LoS'], axis=1)
        #plt.figure(figsize=(20,20))
        #dataplot = sb.heatmap(temp.corr(), cmap="YlGnBu", annot=True)
        #plt.show()

    def Ability(self):
        #self.df.drop_duplicates(inplace=True)
        self.LoS(0.25)
        self.df['HTW'] = 0
        self.df.loc[self.df['LoS'] >= 5, 'HTW'] = 1
        self.df = self.df[self.df['LastTime'] <= 14500]
        #self.df = self.df[self.df['LastTime'] < 8892]
        #self.df = self.df[self.df['LastTime'] > 5075]

        #samp = self.df.sample(n=100000, random_state=1)

        MhtwAbility = self.df.loc[(self.df['HTW'] == 1) & (self.df['Gender'] == 'M'), 'LastTime']
        MallAbility = self.df.loc[self.df['Gender'] == 'M', 'LastTime']
        FhtwAbility = self.df.loc[(self.df['HTW'] == 1) & (self.df['Gender'] == 'F'), 'LastTime']
        FallAbility = self.df.loc[self.df['Gender'] == 'F', 'LastTime']

        #MhtwAbility.drop_duplicates(subset=['Pb'], inplace=True)
        #MallAbility.drop_duplicates(subset=['Pb'], inplace=True)
        #FhtwAbility.drop_duplicates(subset=['Pb'], inplace=True)
        #FallAbility.drop_duplicates(subset=['Pb'], inplace=True)

        #print(str(self.df['Pb'].quantile(q=0.05)))

        Mallhist, medges = np.histogram(MallAbility, bins=np.linspace(3600, 14400, num=10))
        Mhtwhist, _ = np.histogram(MhtwAbility, bins=medges)

        Fallhist, fedges = np.histogram(FallAbility, bins=medges)
        Fhtwhist, _ = np.histogram(FhtwAbility, bins=fedges)

        mhist = []
        for mhtwbin, mallmbin in zip(Mhtwhist, Mallhist):
            mhist.append(mhtwbin / mallmbin)

        fhist = []
        for fhtwbin, fallmbin in zip(Fhtwhist, Fallhist):
            fhist.append(fhtwbin / fallmbin)

        fig, ax1 = plt.subplots(figsize=(10,5))
        ax1.set_xlabel('Previous finish time [Hour:Min]')
        ax1.set_ylabel('Ratio runners hitting the wall')

        x_m = [str(datetime.timedelta(seconds=int(i)))[0:-3] for i in medges[:-1]]
        f_m = [str(datetime.timedelta(seconds=int(i)))[0:-3] for i in fedges[:-1]]

        #x_f = [datetime.time(seconds=i) for i in fedges[:-1]]
        #print(x_m)
        ax1.plot(x_m, mhist, label='Male')
        ax1.plot(x_m, fhist, label='Female')

        plt.legend()
        ax1.grid()

        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/MvsFHtwLastTime.pdf')
        plt.show()

    def ForestRes(self, crossval):
        #plt.rcParams.update(plt.rcParamsDefault)
        dist = 10
        # Preprocess 
        self.LoS(0.25)
        df = pd.DataFrame()
        df = self.df[['5kmPace', '10kmPace', '15kmPace', '20kmPace',
         '21kmPace', 'LoS', 'temperature', 'LastTime', 'LastPaceGroup', 'Gender', 'LastSplitRatio', 'Age', 'Runs']].copy()

        df = pd.get_dummies(df, columns= ['Gender'], drop_first=True)
        df = pd.get_dummies(df, columns= ['LastPaceGroup'])

        # Labeling 
        df['HTW'] = 0
        df.loc[df['LoS'] >= 5, 'HTW'] = 1
        df.drop(['LoS'], axis=1, inplace=True)

        if dist == 5:
          df.drop(['10kmPace', '15kmPace', '20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 10:
          df.drop(['15kmPace', '20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 15:
          df.drop(['20kmPace','21kmPace'], axis=1, inplace=True)
        if dist == 20:
          df.drop(['21kmPace'], axis=1, inplace=True)

        delcols = int(5 - (dist / 5))

        temp = df.drop(['LastPaceGroup_EVEN', 'LastPaceGroup_HTW', 'LastPaceGroup_NEG', 'LastPaceGroup_POS'], axis=1, inplace=True)
        tempdata = df.values
        print(df.columns)
        Xtemp = tempdata[:, 0:8].astype('float32')
        #print(len(Xtemp[0, 0:6]))

        data = df.values
        X = data[:,0:12 - delcols].astype('float32')
        y = data[:,-1].astype('float32')

        clf = joblib.load('/content/drive/MyDrive/Varvetresultat/Models/RandomForestBalancedCV_Age.pkl')
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, stratify=y)
        if crossval == 1:
            oversample = SMOTE()
            over_X, over_y = oversample.fit_resample(X, y)
            over_X_train, over_X_test, over_y_train, over_y_test = train_test_split(over_X, over_y, test_size=0.1, stratify=over_y)

            #Create Stratified K-fold cross validation
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
            scoring = ('f1', 'recall', 'precision')

            #Evaluate SMOTE SRF model
            scores = cross_validate(clf, over_X, over_y, scoring=scoring, cv=cv, n_jobs=-1)
            #Get average evaluation metrics
            print('Mean f1: %.3f' % mean(scores['test_f1']))
            print('Mean recall: %.3f' % mean(scores['test_recall']))
            print('Mean precision: %.3f' % mean(scores['test_precision']))

        X_train, X_test, y_train, y_test = train_test_split(Xtemp, y, test_size=0.1, stratify=y, random_state=1)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        #Create confusion matrix
        fig, ax = plt.subplots()
        #plt.rcParams.update({'font.size': 20})
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Avoid HTW', 'HTW'], ax=ax, cmap='Blues', normalize='all')
        #plt.title('Confusion Matrix')
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/ConfusionMatrixRForest1.pdf')
        plt.show()

        print(metrics.classification_report(y_test, y_pred, target_names=['Avoid HTW', 'HTW']))
        print(dict(zip(df.columns, clf.feature_importances_)))

        print("Calculating Feature importances ...")
        result = permutation_importance(
          clf, X_test, y_test, n_repeats=1, random_state=42, n_jobs=1
        )
        forest_importances = pd.Series(result.importances_mean, index=['5km Pace', '10km Pace', 'Temperature', 'Previous Time', 
                                                                   'Previous Split Ratio', 'Age', 'Runs', 'Gender']) 

        forest_importances.sort_values(ascending=False, inplace=True) # Decending order
        fig, ax = plt.subplots()
        forest_importances.plot.bar(ax=ax, zorder=4)
        #ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        ax.set_xlabel("Feature of a runner")
        plt.grid(zorder=1)
        plt.xticks(rotation=45, ha='right')
        fig.tight_layout()
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/FeatureImportances.pdf')
        plt.show()

        samples =  len(y_test) - 1
        print(f'Number of samples in held out test-set: {samples}')
        '''
        ax = plt.gca()
        rfc_disp = RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax, name='Balanced Random Forest')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/ROC.pdf')
        '''
        low_risk = pd.Series(dtype=np.float32)
        high_risk = pd.Series(dtype=np.float32)
        med_risk = pd.Series(dtype=np.float32)
        false_pos = pd.Series(dtype=np.float32)
        true_pos = pd.Series(dtype=np.float32)

        true_neg = pd.Series(dtype=np.float32)
        false_neg = pd.Series(dtype=np.float32)

        all = pd.Series(dtype=np.float32)

        all_i = [i for i in range(len(y_train))]
        all = df.iloc[all_i,:]

        false_pos_i = [i for i in range(len(y_test)) if (y_pred[i] == 1) and (y_test[i] == 0)]
        false_pos = df.iloc[false_pos_i,:]

        true_pos_i = [i for i in range(len(y_test)) if (y_pred[i] == 1) and (y_test[i] == 1)]
        true_pos = df.iloc[true_pos_i,:]

        true_neg_i = [i for i in range(len(y_test)) if (y_pred[i] == 0) and (y_test[i] == 0)]
        true_neg = df.iloc[true_neg_i,:]

        false_neg_i = [i for i in range(len(y_test)) if (y_pred[i] == 0) and (y_test[i] == 1)]
        false_neg = df.iloc[false_neg_i,:]

        #low_risk_i = [i for i in range(len(y_proba)) if y_proba[i,1] <= 0.25]
        #low_risk = df.iloc[low_risk_i,:]

        #high_risk_i = [i for i in range(len(y_proba)) if y_proba[i,1] >= 0.75]
        #high_risk = df.iloc[high_risk_i,:]

        actual_avoid_i = [i for i in range(len(y_test)) if y_test[i] == 0]
        actual_avoid = df.iloc[actual_avoid_i,:]

        actual_htw_i = [i for i in range(len(y_test)) if y_test[i] == 1]
        actual_htw = df.iloc[actual_htw_i,:]

        pred_avoid_i = [i for i in range(len(y_test)) if y_pred[i] == 0]
        pred_avoid = df.iloc[pred_avoid_i,:]

        pred_htw_i = [i for i in range(len(y_test)) if y_pred[i] == 1]
        pred_htw = df.iloc[pred_htw_i,:]

        nr_false_pos = len(false_pos)
        nr_true_pos = len(true_pos)
        nr_true_neg = len(true_neg)
        nr_false_neg = len(false_neg)

        print(f'False positives: {nr_false_pos}')
        print(f'True positives: {nr_true_pos}')
        print(f'False negatives: {nr_false_neg}')
        print(f'True negatives: {nr_true_neg}')
        print(f'Actual HTW: {len(actual_htw)}')
        print(f'Actual avoid: {len(actual_avoid)}')
        print(f'Pred HTW: {len(pred_htw)}')
        print(f'Pred avoid: {len(pred_avoid)}')
        print(f'How many avoid HTW but predicted as HTW: {nr_false_pos /(nr_false_pos +nr_true_pos)}')
        print(f'How many HTW and predicted as HTW: {nr_true_pos /(nr_false_pos +nr_true_pos)}')
        print(f'How many avoid HTW and predicted as avoid HTW: {nr_true_neg /(nr_true_neg +nr_false_neg)}')
        print(f'How many HTW and predicted as avoid HTW: {nr_false_neg /(nr_true_neg +nr_false_neg)}')

        plt.scatter(x=true_pos['5kmPace'], y=true_pos['10kmPace'], label='True Positive', s=0.5)
        plt.scatter(x=false_pos['5kmPace'], y=false_pos['10kmPace'], label='False Positive', s=0.5)
        plt.legend()

        '''
        plt.figure(figsize=(10,10))
        dataplot = sb.heatmap(wrong_pred.corr(), cmap="YlGnBu", annot=True)
        plt.show()

        plt.hist(right_pred['temperature'])
        plt.show()
        '''

        '''
        names = ['5kmPace', '10kmPace', 'temperature', 'LastTime', 'LastSplitRatio']
        fig, axs = plt.subplots(5, 1)
        fig.set_size_inches(10, 20)
        for ax, name in zip(axs, names):
          if name == '5kmPace' or name == '10kmPace':
            ax.set_xlim(0, 700)
          ax.hist(low_risk[name], bins=50, density=True, histtype='step', color='blue')
          ax.hist(high_risk[name], bins=50, density=True, histtype='step', color='green')
          #ax.hist(false_pos[name], bins=50, density=False, histtype='step', color='red')
          #ax.hist(false_neg[name], bins=50, density=False, histtype='step', color='yellow')
          ax.set_xlabel(name)
          ax.set_ylabel('Runners')
          ax.title.set_text(name)

        plt.tight_layout()
        '''
        '''
        #print(X_train)
        fig, axn = plt.subplots()
        #fig.set_size_inches(10, 10)
        #axn.hist(X_test[:,2], bins=10, density=False, histtype='step', color='blue')
        #axn.hist(X_train[:,2], bins=10, density=False, histtype='step', color='red')
        axn.hist(all['temperature'], bins=10, density=False, histtype='step', color='red')
        '''
        '''
        slows = wrongdpred.loc[wrongdpred["HTW"] == 1, "LastTime"]
        reg = wrongdpred['LastTime']

        plt.rcParams.update({'font.size': 12})
        fig, ax1 = plt.subplots()
        sbins, edges = np.histogram(slows, bins=10)
        regbins, _ = np.histogram(reg, bins=edges)

        fin = []
        for sbin, rbin in zip(sbins, regbins):
          fin.append(sbin / rbin)

        ax1.bar(edges[:-1], fin, width=np.diff(edges), edgecolor="black", align="edge")

        ax1.set_xlabel("Previous finish time [s]")
        ax1.set_ylabel("Ratio of Runners who HTW")
        #plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/Name.pdf')
        plt.show()
        '''
        #df['Split'] = df['10kmPace'] / df['5kmPace']
        #fig, ax = plt.subplots()
        #ax.hist(df['Split'], bins=50, density=False, histtype='step', color='blue')
        #ax.set_xlim(0.8, 1.2)
        #plt.show()
        # Assumes import seaborn as sb

        # Bins are the x and y inputs to the map
        nr_bins = 8

        # To the array containing all input data, add the column of their respective probability. 
        # y_proba is (n_samples, 2) array
        extended_X_test = np.hstack((X_test, np.atleast_2d(y_proba[:, 1]).T))

        # New dataframe is not neccesary but is easier. Add a final column of the output
        # of the heatmap. 
        df_test = pd.DataFrame(extended_X_test, columns=df.columns.delete(6).append(pd.Index(['Prob'])))

        # Bin the x and y values using the same bins, may change heatmap depending on order etc
        binned_5k, edges = np.histogram(X_test[:,0], bins=nr_bins)
        binned_10k, edges_10k = np.histogram(X_test[:,1], bins=edges)

        # Rounding...
        edges = np.round(edges, decimals=0)
        prob = np.zeros(shape=(nr_bins,nr_bins))

        # Ugly but calculate the mean of all values in certain bounds of the bins
        for i in range(nr_bins):
          for j in range(nr_bins):
            prob[i,j] = df_test.loc[(df_test['5kmPace'] >= edges[i]) & (df_test['5kmPace'] < edges[i+1])
                               & (df_test['10kmPace'] >= edges[j]) & (df_test['10kmPace'] < edges[j+1]), 'Prob'].mean()

        # x and y data. x is [[1, 2, 3, ...], [1, 2, 3, ...], ...] y is [[1,1,1, ...], [2, 2, 2, ...], ...]
        # Prob is 'z', the values of the squares in the heatmap
        data = pd.DataFrame({'5 km pace': np.repeat(edges[0:nr_bins], nr_bins), '10 km pace': np.tile(edges[0:nr_bins], nr_bins), 'Prob': prob.flatten()})

        # Magic
        data_pivoted = data.pivot_table(index="10 km pace", columns="5 km pace", values="Prob")

        #Plotting
        plt.figure(figsize=(10,10.5))
        ax = sb.heatmap(data_pivoted, annot=True, cmap='Blues')
        ax.set_title("Predicted Probability of HTW")
        ax.invert_yaxis()
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/Predictions_paces_heatmap.pdf')

        binned_lastTime, edges_lt = np.histogram(X_test[:,3], bins=nr_bins)
        binned_temps, edges_temp = np.histogram(X_test[:,2], bins=nr_bins)

        edges_lt = np.round(edges_lt, decimals=0)
        edges_temp = np.round(edges_temp, decimals=0)

        prob = np.zeros(shape=(nr_bins,nr_bins))

        for i in range(nr_bins):
          for j in range(nr_bins):
            prob[i,j] = df_test.loc[(df_test['LastTime'] >= edges_lt[i]) & (df_test['LastTime'] < edges_lt[i+1])
                               & (df_test['temperature'] >= edges_temp[j]) & (df_test['temperature'] < edges_temp[j+1]), 'Prob'].mean()

        data = pd.DataFrame({'Last Time': np.repeat(edges_lt[0:nr_bins], nr_bins), 'Temperature': np.tile(edges_temp[0:nr_bins], nr_bins), 'Prob': prob.flatten()})

        data_pivoted = data.pivot_table(index="Last Time", columns="Temperature", values="Prob")

        plt.figure(figsize=(10,10.5))
        ax = sb.heatmap(data_pivoted, annot=True, cmap='Blues')
        ax.set_title("Predicted Probability of HTW")
        ax.invert_yaxis()
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/Prediction_LastTime_Temp_heatmap.pdf')

        #plt.show()

        print("Mean values")
        print(df[['5kmPace', '10kmPace', 'temperature', 'LastTime', 'LastSplitRatio', 'Gender_M', 'Age', 'Runs']].mean())
        print("ST-deviations")
        print(df[['5kmPace', '10kmPace', 'temperature', 'LastTime', 'LastSplitRatio', 'Gender_M', 'Age', 'Runs']].std())

        standard_runner = np.array([[330, 332, 17.5, 7286, 1.076427, 1, 40.267222 ,2.651694]]).reshape(1, -1)
        n_samples = 150

        paces_5k = np.linspace(250, 400, n_samples).reshape(n_samples,)
        paces_10k = np.linspace(250, 400, n_samples).reshape(n_samples,)
        temp_range = np.linspace(10, 30, n_samples).reshape(n_samples,)

        standard_runner5 = np.tile(standard_runner, (n_samples,1))
        standard_runner5[:, 0] = paces_5k

        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)
        ax.set_title("Probability with 5:30 10 km pace")
        ax.set_ylabel("Probability to HTW")
        ax.set_xlabel("Relative slowdown during 5-10 km segment")
        ax.scatter(y=clf.predict_proba(standard_runner5)[:,1], x=standard_runner5[:, 1]/standard_runner5[:, 0])

        standard_runner10 = np.tile(standard_runner, (n_samples,1))
        standard_runner10[:, 1] = paces_10k
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)
        ax.set_title("Probability to HTW with 5:30 5 km pace")
        ax.set_ylabel("Probability to HTW")
        ax.set_xlabel("Relative slowdown during 5-10 km segment")
        ax.scatter(y=clf.predict_proba(standard_runner10)[:,1], x=standard_runner10[:, 1]/standard_runner10[:, 0])
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/Prediction_StandRunner_diff10kPaces.pdf')

        standard_runner10 = np.tile(standard_runner, (n_samples,1))
        standard_runner10[:, 2] = temp_range
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 6)
        ax.set_title("Probability to HTW given a standard runner and different temperatures")
        ax.set_ylabel("Probability to HTW")
        ax.set_xlabel("Temperature")
        ax.scatter(y=clf.predict_proba(standard_runner10)[:,1], x=standard_runner10[:, 2])

        plt.show()

    def Bars(self):
        #print(str(self.df['Pb'].quantile(q=0.98)))
        self.df = self.df[self.df['LastTime'] <= 12000]
        #self.df = self.df[self.df['Time'] <= self.df['Pb'].quantile(q=0.98)]
        #self.df = self.df[self.df['Time'] >= self.df['Pb'].quantile(q=0.02)]
        los = 0.25
        self.LoS(los)
        bottom = self.df['LastTime'].min()
        top = self.df['LastTime'].max()
        slows = self.df.loc[self.df["LoS"] >= 5, "LastTime"]
        reg = self.df['LastTime']

        #plt.rcParams.update({'font.size': 12})
        fig, ax1 = plt.subplots()
        sbins, edges = np.histogram(slows, bins=10)
        regbins, _ = np.histogram(reg, bins=edges)

        fin = []
        for sbin, rbin in zip(sbins, regbins):
          fin.append(sbin / rbin)

        ax1.bar(edges[:-1], fin, width=np.diff(edges), edgecolor="black", align="edge")

        ax1.set_xlabel("Previous finish time [s]")
        ax1.set_ylabel("Ratio of Runners who HTW")
        #plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/LastTime.pdf')

        #dnfs = self.df[self.df['Status'] == 'DNF']["Year"]
        #ax2.hist(dnfs, bins=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020])
        plt.show()

    def Basic(self):
        nrsamples = self.df['AthleteId'].count()
        print(f"Number of total samples {nrsamples}")

        nruniquesamples = self.df['AthleteId'].nunique()
        print(f"Number of total samples {nruniquesamples}")

        fig, ax1 = plt.subplots(figsize=(10,5))
        plt.rcParams.update({'font.size': 12})

        ax1.hist(self.df['Year'], bins=np.arange(2010, 2021)-0.5, edgecolor='Black', rwidth=0.8, zorder=5)
        plt.xticks(np.arange(2010, 2020, 1))
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Number of Runners")
        plt.grid(zorder=1)
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/RunnersOverYears.pdf')
        plt.show()
        '''
        #test = pd.DataFrame()
        #test = pd.read_csv(self.directory + '/AllResult.csv')

        tot = 0
        for year in range(2010, 2020):
          temp = pd.read_csv(self.directory + "/" + str(year) + '.csv', header=0, sep=";")
          actual = temp['AthleteId'].count()
          our = self.df.loc[self.df['Year'] == year, 'AthleteId'].count()
          print(str(year) + ": "+ '\n' + str(actual))
          print("Samples in our set: " + str(our))
          tot += actual
          print(temp.isna().any(axis=1).sum())
          print("\n")

        print("total: " + str(tot))
        '''

    def kmeans(self):
        timedata = self.df[["5km", "10km", "15km", "20km", "Time"]].values
        Xtime = timedata[:, 0:5].astype('float32')

        tempdata = self.df[["5kmPace", "10kmPace", "15kmPace", "20kmPace", "21kmPace"]].values
        Xtemp = tempdata[:, 0:5].astype('float32')

        reldata = self.df[["5KmRelativePace", "10KmRelativePace", "15KmRelativePace", "20KmRelativePace", "21KmRelativePace"]].values
        Xrel = reldata[:, 0:5].astype('float32')

        Xrel = Xrel[np.random.choice(Xrel.shape[0], 10000, replace=True), :]

        K=range(2,12)
        wss = []
        '''
        fig, ax = plt.subplots()
        for k in K:
          print(f'k-means for k = {k}')
          dtw_km = TimeSeriesKMeans(n_clusters=k,
                             metric="dtw",
                             verbose=0)
          y_pred = dtw_km.fit_predict(Xtemp)
          wss_iter = dtw_km.inertia_
          wss.append(wss_iter)
        plt.xlabel('K')
        plt.ylabel('Within-Cluster-Sum of Squared Errors')
        plt.plot(K,wss)
        plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/ElbowPlot.pdf')
        plt.show()
        '''
        nr_clusters = 2
        dtw_km = TimeSeriesKMeans(n_clusters=nr_clusters,
                               metric="dtw",
                               max_iter_barycenter=100,
                               verbose=0)
        y_pred = dtw_km.fit_predict(Xrel)

        print(f"Iterations: {dtw_km.n_iter_}")

        colors = plt.cm.rainbow(np.linspace(0, 1, nr_clusters))

        fig1, axs = plt.subplots(nr_clusters, 1)
        fig1.set_size_inches(8, 4)
        for k in range(nr_clusters):
          print(f"Number of runners in cluster {k}: {Xrel[y_pred == k].shape[0]}")
          for member in Xrel[y_pred == k]:
            axs[k].plot(member.ravel(), color=colors[k], alpha=.2)
          axs[k].plot(dtw_km.cluster_centers_[k].ravel(), color='black', alpha=1)
          #axs[k].set_ylim(150, 900)
          axs[k].set_xticks(range(0,5))
        #plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/Clusters7.pdf')
        plt.tight_layout()

        fig2, ax2 = plt.subplots()
        for k in range(nr_clusters):
          ax2.plot(dtw_km.cluster_centers_[k].ravel(), color=colors[k], alpha=1)
        ax2.set_xticks(range(0,5))
        plt.show()

    def Runs(self):
        fig, ax = plt.subplots()
        ax.plot(self.df['Runs'].value_counts())
        print(self.df['Runs'].value_counts())

    def Multimodel(self):
        X, y = self.GetXy()
        rus = RandomUnderSampler(random_state=1, sampling_strategy=0.6)
        ros = RandomOverSampler(random_state=1, sampling_strategy=0.3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
        X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
        X_train_ros_rus, y_train_ros_rus = rus.fit_resample(X_train_ros, y_train_ros)

        X_train_scaled = StandardScaler().fit(X_train_ros_rus).transform(X_train_ros_rus)
        X_test_scaled = StandardScaler().fit(X_test).transform(X_test)

        logisticRegr = LogisticRegression(solver='lbfgs', max_iter=100)
        logisticRegr.fit(X_train_ros_rus, y_train_ros_rus)
        logreg_ypred = logisticRegr.predict(X_test)
        print("Logistic Regression model")
        print(f"Accuracy {logisticRegr.score(X_test, y_test)}")
        print(metrics.classification_report(y_test, logreg_ypred, target_names=['Avoid HTW', 'HTW']))

        linSvm = LinearSVC()
        linSvm.fit(X_train_scaled, y_train_ros_rus)
        linsvm_ypred = linSvm.predict(X_test_scaled)
        print("SVM-linear")
        print(linSvm.score(X_test_scaled, y_test))
        print(metrics.classification_report(y_test, linsvm_ypred, target_names=['Avoid HTW', 'HTW']))

        polySvm = SVC(kernel='poly', gamma=1, max_iter = 1e4)
        polySvm.fit(X_train_scaled, y_train_ros_rus)
        polysvm_ypred = polySvm.predict(X_test_scaled)
        print("SVM-poly")
        print(polySvm.score(X_test_scaled, y_test))
        print(metrics.classification_report(y_test, polysvm_ypred, target_names=['Avoid HTW', 'HTW']))

        rbfSvm = SVC(kernel='rbf', gamma=0.1, max_iter = 1e3)
        rbfSvm.fit(X_train_scaled, y_train_ros_rus)
        rbfsvm_ypred = rbfSvm.predict(X_test_scaled)
        print("SVM-rbf")
        print(rbfSvm.score(X_test_scaled, y_test))
        print(metrics.classification_report(y_test, rbfsvm_ypred, target_names=['Avoid HTW', 'HTW']))
        '''
        X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X, y, test_size=0.1, random_state=1)
        kernel = 1.0 * RBF(1.0)
        gpc = GaussianProcessClassifier(kernel=kernel, random_state=0, n_restarts_optimizer=5).fit(X_train_g, y_train_g)
        print("GaussianPC")
        print(gpc.score(X_test_g, y_test_g))
        '''

        knn = KNeighborsClassifier(3)
        knn.fit(X_train_scaled, y_train_ros_rus)
        knn_ypred = knn.predict(X_test_scaled)
        print("knn")
        print(knn.score(X_test_scaled, y_test))
        print(metrics.classification_report(y_test, knn_ypred, target_names=['Avoid HTW', 'HTW']))

        dtree = DecisionTreeClassifier()
        dtree.fit(X_train_ros_rus, y_train_ros_rus)
        dtree_ypred = dtree.predict(X_test)
        print("Dtree")
        print(dtree.score(X_test, y_test))
        print(metrics.classification_report(y_test, dtree_ypred, target_names=['Avoid HTW', 'HTW']))

        rfr = RandomForestClassifier()
        rfr.fit(X_train_ros_rus, y_train_ros_rus)
        rfr_ypred = rfr.predict(X_test)
        print("rfr")
        print(rfr.score(X_test, y_test))
        print(metrics.classification_report(y_test, rfr_ypred, target_names=['Avoid HTW', 'HTW']))


        pca = PCA(n_components=2)
        pcas = pca.fit_transform(X_train_scaled)
        principalDf = pd.DataFrame(data = pcas, columns = ['comp1', 'comp2'])
        finalDf = pd.concat([principalDf, pd.DataFrame(data=y_train_ros_rus, dtype=int, columns=['target'])], axis = 1)
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel('Component 1', fontsize = 15)
        ax.set_ylabel('Component 2', fontsize = 15)
        ax.set_title('2 Component PCA', fontsize = 20)
        targets = [0,1]
        colors = ['r', 'b']
        for target, color in zip(targets, colors):
          indicesToKeep = finalDf['target'] == target
          ax.scatter(finalDf.loc[indicesToKeep, 'comp1'], finalDf.loc[indicesToKeep, 'comp2'], c=color, s=4)
        ax.legend(targets)
        plt.show()

    def pbs(self):
      df = pd.DataFrame()
      df = self.df[['AthleteId', 'Time']].copy()
      #print(df.groupby('AthleteId')['Time'].head(10))
      print(df.groupby('AthleteId')['Time'].rolling(2).min().head(10))
      return

if __name__ == "__main__":
    Base = Base.PacingProject()
    Base.MakeCSVs()  # Run too make a csv of all races in directory with renamed columns and a smaller with all runners that have completed all races

    # Automatically adds paces, BasePace and DoS. 
    # If ReadSmall then also add Personal Best
    #PacingProject.ReadSmallTestCsv()
    Base.ReadLargeCsv()
    Base.RemoveFaultyData()
    #plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["figure.figsize"] = (8,5)
    #PacingProject.SensitivityPlot()
    #PacingProject.RForest()
    #PacingProject.ForestRes(0)
    #PacingProject.Paceings()
    #PacingProject.Ratios()
    #PacingProject.LossTime()
    PacingProject.Ability()
    #PacingProject.Bars()
    #PacingProject.Runs()
    #PacingProject.Basic()
    #PacingProject.kmeans()
    #PacingProject.Multimodel()
    #PacingProject.pbs()