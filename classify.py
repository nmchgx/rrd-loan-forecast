#! python2
# coding: utf-8
import sys
import pandas as pd
import numpy as np
import datetime
from dateutil import parser
import svm
import rf
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/apply_info_samp.csv')

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

def initData(type=0):
    labels = []
    now = datetime.datetime.now()

    for i in range(len(df)):
        dif = 0
        if (pd.notnull(df['payoff_time'][i])):
            dif = parser.parse(df['payoff_time'][i]) - parser.parse(df['due_date'][i])
            labels.append(converter(dif.days, type=type))
        else:
            labels.append(3)
            # dif = now - parser.parse(df['due_date'][i])
    return labels

def statistics():
    arr1 = []
    arr2 = []
    labels = []
    now = datetime.datetime.now()
    for i in range(len(df)):
        if (pd.notnull(df['payoff_time'][i])):
            dif = parser.parse(df['payoff_time'][i]) - parser.parse(df['due_date'][i])
            arr1.append(dif.days)
        else:
            dif = now - parser.parse(df['due_date'][i])
            arr2.append(dif.days)
        labels.append(dif.days)

    data = pd.DataFrame({'payoff_time':df['payoff_time'],'labels':labels})
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

def classify(labels):
    method = sys.argv[1].upper()

    method_name = {'SVM': 'SVM', 'RF': 'Random Forest'}
    test_size_arr = [0.6, 0.5, 0.4]

    data = df.ix[:, 5:]
    labels_df = pd.DataFrame(data=labels)
    print '方法：%s' % method_name[method]
    for test_size in test_size_arr:
        if (method == 'SVM'):
            train_score, test_score = svm.run(
                data, np.array(labels), test_size)
        elif (method == 'RF'):
            train_score, test_score = rf.run(
                data, np.array(labels), test_size)
        else:
            train_score, test_score = svm.run(
                data, np.array(labels), test_size)

        print '============================================='
        print '训练集 %s | 测试集 %s' % (1 - test_size, test_size)
        print '训练集正确率：%s' % train_score
        print '测试集正确率：%s' % test_score

if __name__ == "__main__":
    statistics()
    labels = initData(type=1)
    classify(labels)