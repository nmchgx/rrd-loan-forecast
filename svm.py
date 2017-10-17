#! python2
# coding: utf-8
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


def run(x, y, test_size=0.6):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=1)

    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    print '============================================='
    print '训练集 %s | 测试集 %s' % (1 - test_size, test_size)
    print '训练集正确率：%s' % clf.score(x_train, y_train)
    print '测试集正确率：%s' % clf.score(x_test, y_test)
