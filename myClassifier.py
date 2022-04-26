from datetime import datetime

import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient
from joblib import Parallel, delayed
from numpy import datetime64
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn_porter import Porter

raw_data_path = 'influxData.csv'
formatted_data_path = 'formattedData_pivot.csv'
mean_and_var_data_path = 'mean_and_var_data.csv'
model_output_path = 'Classifier.js'

label_string_to_number = {'sitting': 0, 'standing': 1, 'walking': 2}
label_number_to_string = {0: 'sitting', 1: 'standing', 2: 'walking'}
value_to_mean = {'alpha': 'alphaMean', 'beta': 'betaMean', 'gamma': 'gammaMean', 'x': 'xMean', 'x0': 'x0Mean',
                 'y': 'yMean', 'y0': 'y0Mean', 'z': 'zMean', 'z0': 'z0Mean'}
value_to_variance = {'alpha': 'alphaVariance', 'beta': 'betaVariance', 'gamma': 'gammaVariance', 'x': 'xVariance',
                     'x0': 'x0Variance', 'y': 'yVariance', 'y0': 'y0Variance', 'z': 'zVariance', 'z0': 'z0Variance'}


# one time use
def getDataFromInflux():
    client = InfluxDBClient(url='https://css21.teco.edu',
                            token='d5oSFVlZ-7TuaJgq4XYosp-6E5Bh_6MsJAit7GbcshHdUh7mKy5v-pFGfH4DGg775t_FwpK7pTsKDItRiM9nJQ==',
                            org='css21')

    queryApi = client.query_api()

    get_devicemotion_query = 'from(bucket: "css21")\
        |> range(start: -365d)\
  |> filter(fn: (r) => r["_measurement"] == "devicemotion")\
  |> filter(fn: (r) => r["_field"] == "alpha" or r["_field"] == "beta" or r["_field"] == "x" or r["_field"] == "gamma" or r["_field"] == "x0" or r["_field"] == "y" or r["_field"] == "y0" or r["_field"] == "z" or r["_field"] == "z0")\
  |> filter(fn: (r) => r["browser"] == "Chrome")\
  |> filter(fn: (r) => r["label"] == "sitting" or r["label"] == "standing" or r["label"] == "walking")'

    return queryApi.query_data_frame(get_devicemotion_query)


def format_data(raw_df):
    # new columns will be ['time', 'subject', 'label', 'alpha', 'beta', 'gamma', 'x', 'x0', 'y', 'y0', 'z', 'z0']
    df_pivot = raw_df.pivot_table(index=['time', 'subject', 'label'], columns=['field'], values='value')
    df_pivot.reset_index(inplace=True)

    # in case pivot changed something
    df_pivot = df_pivot.sort_values(['subject', 'time'])
    return df_pivot


def calc_means_and_vars(formatted_df):
    # to ensure correct mean/var calculation | time: datetime64[ns,UTC]
    data_types_dict = {'subject': str, 'label': str, 'alpha': float, 'beta': float, 'gamma': float,
                       'x': float, 'x0': float, 'y': float, 'y0': float,
                       'z': float, 'z0': float}
    formatted_df = formatted_df.astype(data_types_dict)
    # maps the string label to a corresponding number for classification
    formatted_df['label'] = formatted_df['label'].map(label_string_to_number).astype(int)

    # Calculate average and variance over 1 second buckets and renaming the columns to '-mean'/'-variance'
    mean_df = formatted_df.resample('1S', on='time').mean()
    mean_df = mean_df.dropna().rename(columns=value_to_mean)
    variance_df = formatted_df.resample('1S', on='time').var()
    variance_df = variance_df.dropna().rename(columns=value_to_variance)

    # append both df to one big df
    combined_df = pd.concat([mean_df, variance_df], axis=1)
    combined_df = combined_df.dropna()
    combined_df['label'] = combined_df['label'].astype(int)

    return combined_df


if __name__ == '__main__':
    # 1) I will only load from the database once, and then reuse data from the CSV
    # df = getDataFromInflux()
    # df.to_csv(raw_data_path)

    # 2) let's try loading the raw data from a csv file instead
    # df = pd.read_csv(raw_data_path, dtype=str)
    #
    # df.rename(columns={'_time': 'time', '_value': 'value', '_field': 'field'}, inplace=True)
    # df = df.sort_values(['subject', 'time'])
    #
    # subject_set = set(df['subject'].unique())

    # # opt. 3) for data formatting/processing test purposes, we can make the data set smaller | '0721', '10ad9',
    # test_subjects = ['12f63']
    # df = df[df['subject'].isin(test_subjects)]

    # 4) format the data: moves the 12 rows of values into one row with all corresponding values
    # pivot_df = format_data(df)
    # print(pivot_df)
    # pivot_df.to_csv(formatted_data_path)

    # 5) now calculate the means and variances on the formatted data| removes subject and puts it into time buckets
    # Mean and var are calculated for training, as this reduces the noise of the data immensely.
    # df = pd.read_csv(formatted_data_path, dtype=str, parse_dates=['time'])
    # mean_and_var_df = calc_means_and_vars(df)
    # mean_and_var_df.to_csv(mean_and_var_data_path)

    # 6) Finally: Time to train and verify!
    df = pd.read_csv(mean_and_var_data_path, parse_dates=['time'])
    print(df)
    labels = df['label'].to_numpy().astype(int)
    data = df.drop(labels=['label', 'time'], axis=1).to_numpy()

    # instantiating different models:
    # Models train on mean and variance of params:  'alpha': float, 'beta': float, 'gamma': float,
    #                                               'x': float, 'x0': float, 'y': float, 'y0': float,
    #                                               'z': float, 'z0': float
    np.random.seed(42)  # reproducibility & the answer to everything
    naive_bayes = GaussianNB()  # Gaussian Naive Bayes
    svc = svm.SVC()
    knn = KNeighborsClassifier()  # k is by default 5
    decision_tree = DecisionTreeClassifier()
    multilayer_perceptron = MLPClassifier()
    random_forest = RandomForestClassifier()

    # time to train:
    # naive_bayes.fit(data, labels)
    # svc.fit(data, labels)
    # knn.fit(data, labels)
    # decision_tree.fit(data, labels)
    # multilayer_perceptron.fit(data, labels)
    random_forest.fit(data, labels)

    # evaluate the different classifiers:
    # naive_bayes_score = cross_val_score(naive_bayes, data, labels, cv=10)  # standard=10, as rec in lecture
    # svc_score = cross_val_score(svc, data, labels, cv=10)
    # knn_score = cross_val_score(knn, data, labels, cv=10)
    # decision_tree_score = cross_val_score(decision_tree, data, labels, cv=10)
    # multilayer_perceptron_score = cross_val_score(multilayer_perceptron, data, labels, cv=10)
    random_forest_score = cross_val_score(random_forest, data, labels, cv=10)

    # print('Naive Bayes: ')
    # print(naive_bayes_score)
    # print("%0.2f accuracy with a standard deviation of %0.2f \n" % (naive_bayes_score.mean(), naive_bayes_score.std()))
    #
    # print('SVC: ')
    # print(svc_score)
    # print("%0.2f accuracy with a standard deviation of %0.2f \n" % (svc_score.mean(), svc_score.std()))
    #
    # print('5 Nearest Neighbors: ')
    # print(knn_score)
    # print("%0.2f accuracy with a standard deviation of %0.2f \n" % (knn_score.mean(), knn_score.std()))
    #
    # print('Decision Tree: ')
    # print(decision_tree_score)
    # print("%0.2f accuracy with a standard deviation of %0.2f \n" % (decision_tree_score.mean(), decision_tree_score.std()))
    #
    # print('Multilayer Perceptron: ')
    # print(multilayer_perceptron_score)
    # print("%0.2f accuracy with a standard deviation of %0.2f \n" % (multilayer_perceptron_score.mean(), multilayer_perceptron_score.std()))

    print('Random Forest Score: ')
    print(random_forest_score)
    print("%0.2f accuracy with a standard deviation of %0.2f \n" % (random_forest_score.mean(), random_forest_score.std()))

    # with 1S buckets, default RF was the best with 0.86 mean acc & std of 0.05
    # with 2S buckets, only std increased to 0.06 -> stay with 1S

    porter = Porter(random_forest, language='JS')
    output = porter.export(embed_data=True)
    print(output)
    with open(model_output_path, 'w') as model_text_file:
        model_text_file.write(output)


