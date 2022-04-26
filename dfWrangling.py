import csv

from IPython.core.display_functions import display
from influxdb_client import InfluxDBClient
import pandas as pd
import matplotlib
import matplotlib.backends.backend_tkagg
matplotlib.use('tkagg')

# some options to prettify the output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# our influxdb credentials
token = "2VRzDEtPDnnY1WtVcqgwTuewwHmqYlGiFpDo8PGSwwbY_Tb9f-oj_wIa6qoE9a4u1Y9jPo1mjNWA6JvxIv4fuw=="
org = "css21"
bucket = "css21"
data_path = 'devicemotion_data.csv'



# let's try to get the data
if __name__ == '__main__':
    ## commented out, since the server doesn't seem to respond
    # client = InfluxDBClient(url="https://css21.teco.edu", token=token)
    # query_api = client.query_api()
    # get_devicemotion_query = 'from(bucket: "css21")\
    #                           |> range(start:  2021-04-01T00:00:01Z, stop: now())\
    #                           |> filter(fn: (r) => r["_measurement"] == "devicemotion")\
    #                           |> filter(fn: (r) => r["label"] == "sitting" or r["label"] == "standing" or r["label"] == "walking")\
    #                           |> group(columns: ["label", "subject"])'
    # result = query_api.query(org=org, query=get_devicemotion_query)
    # print("Finished Query")
    # results = []
    # for table in result:
    #     for record in table.records:
    #         results.append((record.get_value(), record.get_field()))
    #
    # print(results)

    # let's try loading the data from a csv file instead
    df = pd.read_csv(data_path)
    df.drop(df.index[[0,1,2]], inplace=True)

    # let's remove fillter columns and start, stop, measurement columns
    df.drop(df.columns[[0, 1, 2, 3, 4, 8]], axis=1, inplace=True)
    df.columns = ['time', 'value', 'field', 'browser', 'label', 'mobile', 'subject']

    # since there seems to be some bad data formatting in the last few rows, we are gonna slice it off
    df = df[:-108]

    # we will create more df for the rows where the browser column is missing
    df2 = pd.read_csv(data_path)
    df2.drop(df2.index[0:466], inplace=True)
    df2.drop(df2.columns[[0, 1, 2, 3, 4, 8, 12]], axis=1, inplace=True)
    df2.columns = ['time', 'value', 'field', 'label', 'mobile', 'subject']

    # to append df2 to df we have to add a column and move it to the right index
    df2 = df2.assign(browser='Unknown')
    browser_column = df2.pop('browser')
    df2.insert(3, 'browser', browser_column)

    # we have some more "title" rows
    df2.drop(df2.index[6:10], inplace=True)
    df2.drop(df2.index[12:16], inplace=True)

    # now append df2 to df
    df = pd.concat([df, df2])

    # group by label and subject
    df.groupby(['label', 'subject'])

    # pivot the field column into multiple columns
    df = df.pivot_table(index=['time', 'subject', 'label'], columns=['field'], values='value')
    #df.reset_index(inplace=True)
    #df = df.set_index("time")
    #df.mean(axis=1).plot(kind='bar')


    display(df)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
