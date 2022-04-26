import pandas as pd
from influxdb_client import InfluxDBClient
from joblib import Parallel, delayed
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
#from sklearn_porter import Porter

valueToMeanMap = {"alpha": "alphaMean", "beta": "betaMean", "gamma": "gammaMean", "x": "xMean", "x0": "x0Mean",
                  "y": "yMean", "y0": "y0Mean", "z": "zMean", "z0": "z0Mean"}

valueToVarianceMap = {"alpha": "alphaVariance", "beta": "betaVariance", "gamma": "gammaVariance", "x": "xVariance",
                      "x0": "x0Variance", "y": "yVariance", "y0": "y0Variance", "z": "zVariance", "z0": "z0Variance"}

classMap = {"standing": 0, "walking": 1, "sitting": 2}

classMapReverse = {0: "standing", 1: "walking", 2: "sitting"}


def getData():
    client = InfluxDBClient(url='https://css21.teco.edu',
                            token='d5oSFVlZ-7TuaJgq4XYosp-6E5Bh_6MsJAit7GbcshHdUh7mKy5v-pFGfH4DGg775t_FwpK7pTsKDItRiM9nJQ==',
                            org='css21')

    queryApi = client.query_api()

    query = 'from(bucket: "css21")\
        |> range(start: -350d)\
  |> filter(fn: (r) => r["_measurement"] == "devicemotion")\
  |> filter(fn: (r) => r["_field"] == "alpha" or r["_field"] == "beta" or r["_field"] == "x" or r["_field"] == "gamma" or r["_field"] == "x0" or r["_field"] == "y" or r["_field"] == "y0" or r["_field"] == "z" or r["_field"] == "z0")\
  |> filter(fn: (r) => r["browser"] == "Chrome")\
  |> filter(fn: (r) => r["label"] == "sitting" or r["label"] == "standing" or r["label"] == "walking")\
  |> filter(fn: (r) => r["mobile"] == "UnknownPhone")\
  |> filter(fn: (r) => r["subject"] == "1453e" or r["subject"] == "1995c" or r["subject"] == "1b190")'

    return queryApi.query_data_frame(query)


def processDataFrames(dataFrame):
    columns = ["subject", "label", "time", "alpha", "beta", "gamma", "x", "x0", "y", "y0", "z", "z0"]
    data = pd.DataFrame(columns=columns)
    time = None
    list = []
    for index, row in dataFrame.iterrows():
        if time is None:
            time = row["time"]
            list.append(row["subject"])
            list.append(row["label"])
            list.append(row["time"])

        if row["time"] != time:
            if len(list) == 12:
                data = data.append(pd.DataFrame([list], columns=columns))

            time = row["time"]
            list = [row["subject"], row["label"], row["time"], row["_value"]]
        else:
            list.append(row["_value"])

            # here data col are: time, subject, label, value
            # with 12 rows, all same time, alpha, then gamma, ...
            # then, when we have a list with 12 variables, we append it ot the dataframe

    return data


def formatData(result):
    result.rename(columns={'_time': 'time'}, inplace=True)
    result = result.sort_values(["subject", "time"])

    subjects = set(result["subject"].unique())
    frames = []

    for subject in subjects:
        frames.append(result.loc[result["subject"] == subject])

    processedFrames = Parallel(n_jobs=3)(delayed(processDataFrames)(data) for data in frames)

    dataframes = pd.DataFrame(
        columns=["alphaMean", "betaMean", "gammaMean", "xMean", "x0Mean", "yMean", "y0Mean", "zMean", "z0Mean",
                 "alphaVariance", "betaVariance", "gammaVariance", "xVariance", "x0Variance", "yVariance",
                 "y0Variance", "zVariance", "z0Variance", "label"])

    for index, data in enumerate(processedFrames):
        data.time = pd.to_datetime(data.time)
        label = data["label"].iloc[0] # get's the label
        if (label == "inUse" or label == "onTable"): # we only want sitting, standing, walking
            continue
        label = classMap[data["label"].iloc[0]] # convert string-label to number

        # Calculate average and variance over 2 second frames and renaming the columns to '-mean'/'-variance'
        data_mean = data.resample("2S", on="time").mean().dropna().rename(columns=valueToMeanMap)
        data_variance = data.resample("2S", on="time").var().dropna().rename(columns=valueToVarianceMap)

        # add to one df as columns?
        data_combined = pd.concat([data_mean, data_variance], axis=1)
        data_combined.loc[:, 'label'] = label # append label column
        data_combined = data_combined.dropna() # drop na's
        dataframes = pd.concat([dataframes, data_combined]) # now append aggregations df

        # data columns are: subject, label, time, alpha, beta... z0
        # dataframes = means, variance, label

    return dataframes


if __name__ == '__main__':
    data = formatData(getData())
    labels = data["label"].to_numpy().astype('int')
    data = data.drop(labels="label", axis=1).to_numpy()
    # we are only training on means and variance, original measurements are no longer involved
    randomForest = RandomForestClassifier(random_state=0)
    nearestNeighbor = KNeighborsClassifier(3)
    SVC = svm.SVC(random_state=0)
    decisionTree = DecisionTreeClassifier(random_state=0)
    naiveBayes = GaussianNB()

    randomForest.fit(data, labels)
    nearestNeighbor.fit(data, labels)
    SVC.fit(data, labels)
    decisionTree.fit(data, labels)
    naiveBayes.fit(data, labels)

    scores = cross_val_score(randomForest, data, labels, cv=10)
    print("Random forest:")
    print(scores)
    scores = cross_val_score(nearestNeighbor, data, labels, cv=10)
    print("3-Nearest Neighbor:")
    print(scores)
    scores = cross_val_score(SVC, data, labels, cv=10)
    print("SVC:")
    print(scores)
    scores = cross_val_score(decisionTree, data, labels, cv=10)
    print("Decision Tree:")
    print(scores)
    scores = cross_val_score(naiveBayes, data, labels, cv=10)
    print("Naive Bayes:")
    print(scores)

    #porter = Porter(randomForest, language='JS')
    #output = porter.export(embed_data=True)
    #with open("RandomForestClassifier.js", "w") as text_file:
    #    text_file.write(output)
