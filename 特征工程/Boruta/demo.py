# -*- coding: utf-8 -*-
"""
@author: wangkang (*/ω＼*)
@file: demo.py
@time: 2019/2/14 14:28
@desc: Boruta 进行特征选择
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

# load X and y
# NOTE BorutaPy accepts numpy arrays only, hence the .values attribute
X = pd.read_csv('examples/test_X.csv', index_col=0).values
y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
y = y.ravel()

# define random forest classifier, with utilising all cores and
# sampling in proportion to y labels
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

# define Boruta feature selection method
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

# find all relevant features - 5 features should be selected
feat_selector.fit(X, y)

# check selected features - first 5 features are selected
print('feat_selector.support_ ：', feat_selector.support_)

# check ranking of features
print('feat_selector.ranking_ :', feat_selector.ranking_)

print('n_features_ :',feat_selector.n_features_)
# call transform() on X to filter it down to selected features
X_filtered = feat_selector.transform(X)
pass