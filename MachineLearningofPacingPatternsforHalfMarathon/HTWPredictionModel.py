import argparse

import numpy as np
import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def main(args):
    df = pd.DataFrame()
    df = pd.read_csv("../Varvetresultat/HTWPrevTime2.csv")
    if args.multimodel:
        multimodel(df)
    if args.train and not args.train.isspace():
        randomForest(df, args.train)
    if args.results:
        forestRes(df, args.results)


def getXy(df_orig):
    dist = 10

    df = pd.DataFrame()
    df = df_orig[['5kmPace', '10kmPace', '15kmPace', '20kmPace',
                  '21kmPace', 'temperature', 'LastTime',
                  'Gender', 'LastSplitRatio', 'Age', 'Runs', 'SplitRatio', 'HTW']].copy()

    df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    if dist == 0:
        df.drop(['5kmPace', '10kmPace', '15kmPace',
                '20kmPace', '21kmPace'], axis=1, inplace=True)
    if dist == 5:
        df.drop(['10kmPace', '15kmPace', '20kmPace',
                '21kmPace'], axis=1, inplace=True)
    if dist == 10:
        df.drop(['15kmPace', '20kmPace', '21kmPace'], axis=1, inplace=True)
    if dist == 15:
        df.drop(['20kmPace', '21kmPace'], axis=1, inplace=True)
    if dist == 20:
        df.drop(['21kmPace'], axis=1, inplace=True)

    delcols = int(5 - (dist / 5))

    df['Class'] = 'Normal'
    df.loc[df['SplitRatio'] < 1, 'Class'] = 'Negative Split'
    df.loc[df['HTW'] == 1, 'Class'] = 'HTW'

    df.drop(['HTW'], axis=1, inplace=True)
    df.drop(['SplitRatio'], axis=1, inplace=True)

    print(df.columns)

    X = df.values[:, 0:-2].astype('float32')
    y = df.values[:, -1]
    return (X, y)


def randomForest(df, model_name):
    X, y = getXy(df)
    #compute_sample_weight('balanced', np.unique(df.y), df.y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1)
    if (True):
        custom_f1 = make_scorer(
            f1_score, greater_is_better=True, average="weighted", labels=['Negative Split', 'Normal', 'HTW'])
        # Number of trees in random forest
        n_estimators = [int(x)
                        for x in np.linspace(start=100, stop=1000, num=10)]
        # Number of features to consider at every split
        max_features = ['sqrt', 'log2']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 200, num=10)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10, 15, 20]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [int(x) for x in np.linspace(1, 20, num=10)]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        criterion = ['gini', 'entropy']
        #classweight = [{0:1, 1: w} for w in [int(x) for x in np.linspace(0.5, 100, num = 20)]]
        sampling_strat = ['majority', 'not minority', 'not majority', 'all']
        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap,
                       'criterion': criterion,
                       'sampling_strategy': sampling_strat}
        # 'class_weight': classweight}
        # sampling_strategy= 0.5, class_weight='balanced' n_estimators=150, random_state=0, class_weight={0:1, 1:10}
        clf = BalancedRandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                                       n_iter=10, cv=3, verbose=4, random_state=42, scoring=custom_f1, n_jobs=4)
        rf_random.fit(X_train, y_train)
        print(rf_random.best_params_)
        best_est = rf_random.best_estimator_
        best_est.fit(X_train, y_train)
        y_pred = best_est.predict(X_test)
    #clf = BalancedRandomForestClassifier(verbose=True)
    #print("Fitting the model ...")
    #params1 = {'n_estimators': 1200, 'min_samples_split': 20, 'min_samples_leaf': 11, 'max_features': 'sqrt', 'max_depth': 110, 'criterion': 'gini', 'bootstrap': True}
    #params_age = {'n_estimators': 2000, 'min_samples_split': 10, 'min_samples_leaf': 15, 'max_features': 'log2', 'max_depth': 170, 'criterion': 'entropy', 'bootstrap': False}
    # n_estimators=2000, min_samples_split=10, min_samples_leaf=15, max_features='log2', max_depth=170, criterion='entropy', bootstrap=False
    #clf.fit(X_train, y_train)
    y_pred = best_est.predict(X_test)
    fig, ax = plt.subplots()
    # display_labels=['Neg Split', 'Normals', 'HTW']
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap='Blues', normalize=None)
    plt.savefig('confmatrix.pdf', format='pdf')
    plt.show()

    print(metrics.classification_report(y_test, y_pred))
    features = ['5kmPace', '10kmPace', 'temperature', 'LastTime', 'LastSplitRatio',
                'Age', 'Runs', 'SplitRatio', 'Gender_M']
    print(dict(zip(features, best_est.feature_importances_)))
    joblib.dump(best_est, f'./Models/{model_name}.pkl')


def forestRes(df, model_name):
    X, y = getXy(df)

    clf = joblib.load(f'./Models/{model_name}.pkl')

    _, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=1)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    # Create confusion matrix
    fig, ax = plt.subplots()
    #plt.rcParams.update({'font.size': 20})
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, ax=ax, cmap='Blues', normalize='all')
    #plt.title('Confusion Matrix')
    # plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/ConfusionMatrixRForest1.pdf')
    plt.show()

    print(metrics.classification_report(y_test, y_pred))
    print(dict(zip(df.columns, clf.feature_importances_)))
    return 0
    print("Calculating Feature importances ...")
    result = permutation_importance(
        clf, X_test, y_test, n_repeats=1, random_state=42, n_jobs=1
    )
    forest_importances = pd.Series(result.importances_mean, index=['5km Pace', '10km Pace', 'Temperature', 'Previous Time',
                                                                   'Previous Split Ratio', 'Age', 'Runs', 'Gender'])

    forest_importances.sort_values(
        ascending=False, inplace=True)  # Decending order
    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax, zorder=4)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    ax.set_xlabel("Feature of a runner")
    plt.grid(zorder=1)
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    # plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/FeatureImportances.pdf')
    plt.show()

    samples = len(y_test) - 1
    print(f'Number of samples in held out test-set: {samples}')
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        clf, X_test, y_test, ax=ax, name='Balanced Random Forest')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    # plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/ROC.pdf')

    false_pos = pd.Series(dtype=np.float32)
    true_pos = pd.Series(dtype=np.float32)

    true_neg = pd.Series(dtype=np.float32)
    false_neg = pd.Series(dtype=np.float32)

    all = pd.Series(dtype=np.float32)

    all_i = [i for i in range(len(y_train))]
    all = df.iloc[all_i, :]

    false_pos_i = [i for i in range(len(y_test)) if (
        y_pred[i] == 1) and (y_test[i] == 0)]
    false_pos = df.iloc[false_pos_i, :]

    true_pos_i = [i for i in range(len(y_test)) if (
        y_pred[i] == 1) and (y_test[i] == 1)]
    true_pos = df.iloc[true_pos_i, :]

    true_neg_i = [i for i in range(len(y_test)) if (
        y_pred[i] == 0) and (y_test[i] == 0)]
    true_neg = df.iloc[true_neg_i, :]

    false_neg_i = [i for i in range(len(y_test)) if (
        y_pred[i] == 0) and (y_test[i] == 1)]
    false_neg = df.iloc[false_neg_i, :]

    actual_avoid_i = [i for i in range(len(y_test)) if y_test[i] == 0]
    actual_avoid = df.iloc[actual_avoid_i, :]

    actual_htw_i = [i for i in range(len(y_test)) if y_test[i] == 1]
    actual_htw = df.iloc[actual_htw_i, :]

    pred_avoid_i = [i for i in range(len(y_test)) if y_pred[i] == 0]
    pred_avoid = df.iloc[pred_avoid_i, :]

    pred_htw_i = [i for i in range(len(y_test)) if y_pred[i] == 1]
    pred_htw = df.iloc[pred_htw_i, :]

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
    print(
        f'How many avoid HTW but predicted as HTW: {nr_false_pos /(nr_false_pos +nr_true_pos)}')
    print(
        f'How many HTW and predicted as HTW: {nr_true_pos /(nr_false_pos +nr_true_pos)}')
    print(
        f'How many avoid HTW and predicted as avoid HTW: {nr_true_neg /(nr_true_neg +nr_false_neg)}')
    print(
        f'How many HTW and predicted as avoid HTW: {nr_false_neg /(nr_true_neg +nr_false_neg)}')

    plt.scatter(x=true_pos['5kmPace'],
                y=true_pos['10kmPace'], label='True Positive', s=0.5)
    plt.scatter(x=false_pos['5kmPace'],
                y=false_pos['10kmPace'], label='False Positive', s=0.5)
    plt.legend()

    # Bins are the x and y inputs to the map
    nr_bins = 8

    # To the array containing all input data, add the column of their respective probability.
    # y_proba is (n_samples, 2) array
    extended_X_test = np.hstack((X_test, np.atleast_2d(y_proba[:, 1]).T))

    # New dataframe is not neccesary but is easier. Add a final column of the output
    # of the heatmap.
    df_test = pd.DataFrame(extended_X_test, columns=df.columns.delete(
        6).append(pd.Index(['Prob'])))

    # Bin the x and y values using the same bins, may change heatmap depending on order etc
    binned_5k, edges = np.histogram(X_test[:, 0], bins=nr_bins)
    binned_10k, edges_10k = np.histogram(X_test[:, 1], bins=edges)

    # Rounding...
    edges = np.round(edges, decimals=0)
    prob = np.zeros(shape=(nr_bins, nr_bins))

    # Ugly but calculate the mean of all values in certain bounds of the bins
    for i in range(nr_bins):
        for j in range(nr_bins):
            prob[i, j] = df_test.loc[(df_test['5kmPace'] >= edges[i]) & (df_test['5kmPace'] < edges[i+1])
                                     & (df_test['10kmPace'] >= edges[j]) & (df_test['10kmPace'] < edges[j+1]), 'Prob'].mean()

    # x and y data. x is [[1, 2, 3, ...], [1, 2, 3, ...], ...] y is [[1,1,1, ...], [2, 2, 2, ...], ...]
    # Prob is 'z', the values of the squares in the heatmap
    data = pd.DataFrame({'5 km pace': np.repeat(edges[0:nr_bins], nr_bins), '10 km pace': np.tile(
        edges[0:nr_bins], nr_bins), 'Prob': prob.flatten()})

    # Magic
    data_pivoted = data.pivot_table(
        index="10 km pace", columns="5 km pace", values="Prob")

    # Plotting
    plt.figure(figsize=(10, 10.5))
    ax = sb.heatmap(data_pivoted, annot=True, cmap='Blues')
    ax.set_title("Predicted Probability of HTW")
    ax.invert_yaxis()
    # plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/Predictions_paces_heatmap.pdf')

    binned_lastTime, edges_lt = np.histogram(X_test[:, 3], bins=nr_bins)
    binned_temps, edges_temp = np.histogram(X_test[:, 2], bins=nr_bins)

    edges_lt = np.round(edges_lt, decimals=0)
    edges_temp = np.round(edges_temp, decimals=0)

    prob = np.zeros(shape=(nr_bins, nr_bins))

    for i in range(nr_bins):
        for j in range(nr_bins):
            prob[i, j] = df_test.loc[(df_test['LastTime'] >= edges_lt[i]) & (df_test['LastTime'] < edges_lt[i+1])
                                     & (df_test['temperature'] >= edges_temp[j]) & (df_test['temperature'] < edges_temp[j+1]), 'Prob'].mean()

    data = pd.DataFrame({'Last Time': np.repeat(edges_lt[0:nr_bins], nr_bins), 'Temperature': np.tile(
        edges_temp[0:nr_bins], nr_bins), 'Prob': prob.flatten()})

    data_pivoted = data.pivot_table(
        index="Last Time", columns="Temperature", values="Prob")

    plt.figure(figsize=(10, 10.5))
    ax = sb.heatmap(data_pivoted, annot=True, cmap='Blues')
    ax.set_title("Predicted Probability of HTW")
    ax.invert_yaxis()
    # plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/Prediction_LastTime_Temp_heatmap.pdf')

    # plt.show()

    print("Mean values")
    print(df[['5kmPace', '10kmPace', 'temperature', 'LastTime',
          'LastSplitRatio', 'Gender_M', 'Age', 'Runs']].mean())
    print("ST-deviations")
    print(df[['5kmPace', '10kmPace', 'temperature', 'LastTime',
          'LastSplitRatio', 'Gender_M', 'Age', 'Runs']].std())

    standard_runner = np.array(
        [[330, 332, 17.5, 7286, 1.076427, 1, 40.267222, 2.651694]]).reshape(1, -1)
    n_samples = 150

    paces_5k = np.linspace(250, 400, n_samples).reshape(n_samples,)
    paces_10k = np.linspace(250, 400, n_samples).reshape(n_samples,)
    temp_range = np.linspace(10, 30, n_samples).reshape(n_samples,)

    standard_runner5 = np.tile(standard_runner, (n_samples, 1))
    standard_runner5[:, 0] = paces_5k

    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_title("Probability with 5:30 10 km pace")
    ax.set_ylabel("Probability to HTW")
    ax.set_xlabel("Relative slowdown during 5-10 km segment")
    ax.scatter(y=clf.predict_proba(standard_runner5)[
               :, 1], x=standard_runner5[:, 1]/standard_runner5[:, 0])

    standard_runner10 = np.tile(standard_runner, (n_samples, 1))
    standard_runner10[:, 1] = paces_10k
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_title("Probability to HTW with 5:30 5 km pace")
    ax.set_ylabel("Probability to HTW")
    ax.set_xlabel("Relative slowdown during 5-10 km segment")
    ax.scatter(y=clf.predict_proba(standard_runner10)[
               :, 1], x=standard_runner10[:, 1]/standard_runner10[:, 0])
    # plt.savefig('/content/drive/MyDrive/Varvetresultat/Figs/Prediction_StandRunner_diff10kPaces.pdf')

    standard_runner10 = np.tile(standard_runner, (n_samples, 1))
    standard_runner10[:, 2] = temp_range
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)
    ax.set_title(
        "Probability to HTW given a standard runner and different temperatures")
    ax.set_ylabel("Probability to HTW")
    ax.set_xlabel("Temperature")
    ax.scatter(y=clf.predict_proba(standard_runner10)
               [:, 1], x=standard_runner10[:, 2])

    plt.show()


def multimodel(df):
    X, y = getXy(df)
    rus = RandomUnderSampler(random_state=1, sampling_strategy=0.6)
    ros = RandomOverSampler(random_state=1, sampling_strategy=0.3)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    X_train_ros_rus, y_train_ros_rus = rus.fit_resample(
        X_train_ros, y_train_ros)

    X_train_scaled = StandardScaler().fit(X_train_ros_rus).transform(X_train_ros_rus)
    X_test_scaled = StandardScaler().fit(X_test).transform(X_test)

    logisticRegr = LogisticRegression(solver='lbfgs', max_iter=100)
    logisticRegr.fit(X_train_scaled, y_train_ros_rus)
    logreg_ypred = logisticRegr.predict(X_test_scaled)
    print("Logistic Regression model")
    print(f"Accuracy {logisticRegr.score(X_test, y_test)}")
    print(metrics.classification_report(
        y_test, logreg_ypred, target_names=['Avoid HTW', 'HTW']))

    print("\n----------------------------\n")

    linSvm = LinearSVC(max_iter=1e5)
    linSvm.fit(X_train_scaled, y_train_ros_rus)
    linsvm_ypred = linSvm.predict(X_test_scaled)
    print("SVM-linear")
    print(linSvm.score(X_test_scaled, y_test))
    print(metrics.classification_report(
        y_test, linsvm_ypred, target_names=['Avoid HTW', 'HTW']))

    print("\n----------------------------\n")

    polySvm = SVC(kernel='poly', gamma=1, max_iter=1e4)
    polySvm.fit(X_train_scaled, y_train_ros_rus)
    polysvm_ypred = polySvm.predict(X_test_scaled)
    print("SVM-poly")
    print(polySvm.score(X_test_scaled, y_test))
    print(metrics.classification_report(
        y_test, polysvm_ypred, target_names=['Avoid HTW', 'HTW']))
    return 0
    rbfSvm = SVC(kernel='rbf', gamma=0.1, max_iter=1e3)
    rbfSvm.fit(X_train_scaled, y_train_ros_rus)
    rbfsvm_ypred = rbfSvm.predict(X_test_scaled)
    print("SVM-rbf")
    print(rbfSvm.score(X_test_scaled, y_test))
    print(metrics.classification_report(
        y_test, rbfsvm_ypred, target_names=['Avoid HTW', 'HTW']))
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
    print(metrics.classification_report(
        y_test, knn_ypred, target_names=['Avoid HTW', 'HTW']))

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train_ros_rus, y_train_ros_rus)
    dtree_ypred = dtree.predict(X_test)
    print("Dtree")
    print(dtree.score(X_test, y_test))
    print(metrics.classification_report(
        y_test, dtree_ypred, target_names=['Avoid HTW', 'HTW']))

    rfr = RandomForestClassifier()
    rfr.fit(X_train_ros_rus, y_train_ros_rus)
    rfr_ypred = rfr.predict(X_test)
    print("rfr")
    print(rfr.score(X_test, y_test))
    print(metrics.classification_report(
        y_test, rfr_ypred, target_names=['Avoid HTW', 'HTW']))

    pca = PCA(n_components=2)
    pcas = pca.fit_transform(X_train_scaled)
    principalDf = pd.DataFrame(data=pcas, columns=['comp1', 'comp2'])
    finalDf = pd.concat([principalDf, pd.DataFrame(
        data=y_train_ros_rus, dtype=int, columns=['target'])], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Component 1', fontsize=15)
    ax.set_ylabel('Component 2', fontsize=15)
    ax.set_title('2 Component PCA', fontsize=20)
    targets = [0, 1]
    colors = ['r', 'b']
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'comp1'],
                   finalDf.loc[indicesToKeep, 'comp2'], c=color, s=4)
    ax.legend(targets)
    plt.show()


if __name__ == "__main__":
    # TODO Insert link to paper
    parser = argparse.ArgumentParser(
        description='Tool to replicate results of the paper Machine Learning of Pacing Patterns for Half Marathon.')
    parser.add_argument('--train', '-t',
                        default='',
                        type=str,
                        help='Retrain model, provide name of model (appears in /Models)')
    parser.add_argument('--results', '-r',
                        default='',
                        type=str,
                        help='Results of a provided model name (automatically appends folder name)')
    parser.add_argument('--multimodel', '-m',
                        action='store_true',
                        help='Trains multiple classification models (about 10) and ouputs performance metrics.')
    args = parser.parse_args()
    main(args)
