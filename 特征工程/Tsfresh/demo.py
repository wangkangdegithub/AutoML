# -*- coding: utf-8 -*-
"""
@author: wangkang (*/ω＼*)
@file: demo.py
@time: 2019/2/14 15:07
@desc:
"""

import matplotlib.pylab as plt

from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

if __name__ == '__main__':
    N = 500
    df = pd.read_csv('UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt', delim_whitespace=True, header=None)
    y = pd.read_csv('UCI HAR Dataset/train/y_train.txt', delim_whitespace=True, header=None, squeeze=True)[:N]

    # plt.title('accelerometer reading')
    # plt.plot(df.ix[0, :])
    # plt.show()

    #
    extraction_settings = ComprehensiveFCParameters()
    master_df = pd.DataFrame({'feature': df[:N].values.flatten(),
                              'id': np.arange(N).repeat(df.shape[1])})

    # 时间序列特征工程
    X = extract_features(timeseries_container=master_df, n_jobs=0, column_id='id', impute_function=impute,
                         default_fc_parameters=extraction_settings)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    cl = DecisionTreeClassifier()
    cl.fit(X_train, y_train)
    print(classification_report(y_test, cl.predict(X_test)))

    # 未进行时间序列特征工程
    X_1 = df.ix[:N - 1, :]
    X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=.2)
    cl = DecisionTreeClassifier()
    cl.fit(X_train, y_train)
    print(classification_report(y_test, cl.predict(X_test)))

    # 此外，对于进行时间序列特征工程后的数据集进行特征选择，进一步提高模型指标

    # 利用tsfresh.select_features方法进行特征选择，然而，由于其仅适用于二进制分类或回归任务。
    # 对于6个标签的多分类，我们将选择问题分解为6个二元分类问题。对于每一种，我们都可以进行二元分类特征选择:
    relevant_features = set()
    for label in y.unique():
        y_train_binary = y_train == label
        X_train_filtered = select_features(X_train, y_train_binary)
        print("Number of relevant features for class {}: {}/{}".format(label, X_train_filtered.shape[1],
                                                                       X_train.shape[1]))
        relevant_features = relevant_features.union(set(X_train_filtered.columns))
    X_train_filtered = X_train[list(relevant_features)]
    X_test_filtered = X_test[list(relevant_features)]
    cl = DecisionTreeClassifier()
    cl.fit(X_train_filtered, y_train)
    print(classification_report(y_test, cl.predict(X_test_filtered)))

    pass
