#! python2
# coding: utf-8
import pandas as pd
import numpy as np
import datetime
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns
plt.switch_backend('agg')


def converter(data, type=0):
    if type == 0:
        # 0:<10days | 1: >10days
        if (data <= 10):
            return 0
        else:
            return 1
    elif type == 1:
        # 0:<12days | 1: 10-30days | 2: >30days | 3:
        if (data <= 10):
            return 0
        elif (data > 10 and data <= 30):
            return 1
        else:
            return 2
    return data


def getLabels(data, type=0):
    labels = []
    now = datetime.datetime.now()

    for i in range(len(data)):
        dif = 0
        if (pd.notnull(data['payoff_time'][i])):
            labels.append(0)
            # dif = parser.parse(data['payoff_time'][i]) - \
            #     parser.parse(data['due_date'][i])
            # labels.append(converter(dif.days, type=type))
        else:
            labels.append(1)
    return np.array(labels)


def statistics(data):
    arr1 = []
    arr2 = []
    labels = []
    now = datetime.datetime.now()
    for i in range(len(data)):
        if (pd.notnull(data['payoff_time'][i])):
            dif = parser.parse(data['payoff_time'][i]) - \
                parser.parse(data['due_date'][i])
            arr1.append(dif.days)
        else:
            dif = now - parser.parse(data['due_date'][i])
            arr2.append(dif.days)
        labels.append(dif.days)

    data = pd.DataFrame({'payoff_time': data['payoff_time'], 'labels': labels})
    data.to_csv('output/test.csv')

    sns.distplot(arr1, label='arr1')
    plt.title('arr1')
    plt.savefig('output/arr1.png')
    plt.close()
    sns.distplot(arr2, label='arr1')
    plt.title('arr2')
    plt.savefig('output/arr2.png')
    plt.close()
    print 'arr1: Max: %s Min: %s' % (max(arr1), min(arr1))
    print 'arr2: Max: %s Min: %s' % (max(arr2), min(arr2))



def main():
    path = '/opt/thucollect/data/'
    application_info = pd.read_csv(path + 'application_info.csv', index_col='loan_id', low_memory=False)
    coll_perf = pd.read_csv(path + 'coll_perf.csv', low_memory=False)
    # data
    df = pd.merge(coll_perf, application_info.loc[:, :'age'], left_on='loan_id', right_index=True)
    labels = getLabels(df, type=1)
    statistics(df)
    data = df.drop('user_key', 1)
    data = df.loc[:, 'is_retry':].replace(
        {True: 1, False: 0, 'True': 1, 'False': 0, 'Invalid': -1, 'NULL':-1, 'F': 0, 'M': 1, np.nan: -1})

    np.save('data/data_app.npy', np.array(data.values, dtype='float32'))
    np.save('data/labels.npy', labels)



if __name__ == "__main__":
    main()
