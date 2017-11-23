#! python2
# coding: utf-8
import sys
import numpy as np
import svm
import rf


def loan_type(s):
    it = {'1': 0, '2': 1, '3': 0, '4': 1, '5': 2, '6': 0, '7': 0}
    return it[s]


if __name__ == "__main__":
    loan_2 = 'data/loan_2.csv'
    sourceData = np.loadtxt(loan_2, dtype=float, delimiter=',',
                            converters={0: loan_type}, skiprows=1)

    # x是数据 y是标签
    y, x = np.split(sourceData, (1,), axis=1)

    method = sys.argv[1].upper()

    method_name = {'SVM': 'SVM', 'RF': 'Random Forest'}
    test_size_arr = [0.6, 0.5, 0.4]

    print '方法：%s' % method_name[method]
    for test_size in test_size_arr:
        if (method == 'SVM'):
            train_score, test_score = svm.run(
                x, y, test_size)
        elif (method == 'RF'):
            train_score, test_score = rf.run(
                x, y, test_size)
        else:
            train_score, test_score = svm.run(
                x, y, test_size)

        print '============================================='
        print '训练集 %s | 测试集 %s' % (1 - test_size, test_size)
        print '训练集正确率：%s' % train_score
        print '测试集正确率：%s' % test_score
