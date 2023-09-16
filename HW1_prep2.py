import pandas as pd
import numpy as np

data = pd.read_csv('HW1/covid.train.csv')
x = data[data.columns[1:94]] # data.columns[1:94]取出第1~93個欄位的名字
y = data[data.columns[94]]

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

fit = SelectKBest(score_func=f_regression, k=15).fit(x,y)
dfscores = pd.DataFrame(fit.scores_)
dfpvalues = pd.DataFrame(fit.pvalues_)
dfcolumns = pd.DataFrame(x.columns)
# concat two dataframes for better visualization 
# featureScores = pd.concat([dfcolumns,dfscores],axis=0) 若寫0會變成兩個向量接起來變一個向量
featureScores = pd.concat([dfcolumns,dfscores,dfpvalues],axis=1)
featureScores.columns = ['Specs','Score','P-value']  # naming the dataframe columns
print(featureScores.nlargest(15,'Score'))  # print 15 best features
print(featureScores.nsmallest(15,'P-value'))
print(list(fit.get_support(True))) # 用get_support(True)可以直接取得哪些k是分數最高的

# 或是要這樣取index也可以
index=featureScores.nlargest(15,'Score').index
feats=[]
for i in index:
    feats.append(i) # 取出features所在index
print(feats)

# 參考連結:https://zhuanlan.zhihu.com/p/141506312