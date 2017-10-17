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
    loan_2 = 'loan_2.csv'
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
            svm.run(x, y, test_size)
        elif (method == 'RF'):
            rf.run(x, y, test_size)
