#! python2
# coding: utf-8
import sys
import numpy as np
import svm
import rf

def classify(data, labels):
    method = sys.argv[1].upper()

    method_name = {'SVM': 'SVM', 'RF': 'Random Forest'}
    test_size_arr = [0.6, 0.5, 0.4]

    print '方法：%s' % method_name[method]
    for test_size in test_size_arr:
        if (method == 'SVM'):
            train_score, test_score = svm.run(data, labels, test_size)
        elif (method == 'RF'):
            train_score, test_score = rf.run(data, labels, test_size)
        else:
            train_score, test_score = svm.run(data, labels, test_size)

        print '============================================='
        print '训练集 %s | 测试集 %s' % (1 - test_size, test_size)
        print '训练集正确率：%s' % train_score
        print '测试集正确率：%s' % test_score


def main():
    path = 'data/'
    data = np.load(path+'data_app.npy')
    labels = np.load(path+'labels.npy')
    classify(data, labels)


if __name__ == "__main__":
    main()
