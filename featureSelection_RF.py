#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 23:02:30 2017

@author: hanmufu
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score

#数据处理
data = pd.read_csv('C:/Users/YAO/Documents/taggedData(2).csv')
y=data.label1 
list=['name','code','area','industry']
x=data.drop(list,axis = 1 )

#分成训练数据和测试数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#训练
clf_rf = RandomForestClassifier(random_state=0)      
clr_rf = clf_rf.fit(x_train,y_train)


#正确率
ac = accuracy_score(y_test,clf_rf.predict(x_test))
print('Accuracy is: ',ac)

#热点图
cm = confusion_matrix(y_test,clf_rf.predict(x_test))
sns.heatmap(cm,annot=True,fmt="d")