import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt

data_path="covid.train.csv"
data=pd.read_csv(data_path)
# print(data.head())
print(data.info())
# print(data.isnull().sum()) # 告訴你每個columns有幾個null data

# 迴歸分析: 探討資料間因果關係，發生機率的一種統計方法or做預測在資料探勘中
# descriptive statistics:敘述統計，反應資料真實狀況(次數分配)
# inferential statistics:推論統計，透過test方法讓樣本能推論母體(變異數/迴歸分析)
print(data.describe().iloc[:,41:59]) # 可畫出各欄位的敘述性統計(平均數、標準差、百分位數等)

dataf=data.iloc[:,list(range(41))+[58,76]] # 選第0~39個欄位再加第58、76個

# dataf.iloc[:,1:]=data.iloc[:,1:41].astype("category") 一次改前40個欄位都變成類別

state=list(dataf.columns[1:41])

dataf.insert(41,"days",dataf.groupby(state).cumcount()) # 在第41個維度插入新的欄位，並以AL、AK等40個地區分組，計算1出現的次數從0~group-1，就算出每個地區第0天~第group-1天
# dataf.loc[:101,"days"] 用loc可選擇某欄位某幾筆

data_tpmean=dataf.groupby(state)["tested_positive"].mean()
data_tpstd=dataf.groupby(state)["tested_positive"].std()
data_list=pd.concat([data_tpmean,data_tpstd],axis=1).values.tolist() # 把兩個pd series合併成一個dataframe再轉成list，就可以變成兩兩一對的形式了
plt.plot(state, data_list)
plt.show() # 從圖可看出不同地區的陽性比例差很多

plt.scatter(data.loc[:,"tested_positive"], data.loc[:,"tested_positive.1"]) # 肉眼去看某兩個變數的相關性
plt.show()

feats=list(data.iloc[:,41:].corr().sort_values("tested_positive.2",ascending=False).iloc[:,-1][1:15].index)
columns=pd.Series(list(data.columns))
index=[data.iloc[:,1:].columns.get_loc(col) for col in feats]
print(index) # [75, 57, 42, 60, 78, 43, 61, 79, 40, 58, 76, 41, 59, 77]

plt.figure(figsize=(10, 6))
for group in state: # 把每個state的days對tested_positive的圖都畫出來
    dataG=dataf[dataf[group]==1]
    plt.plot(dataG["days"],dataG['tested_positive'],label=f'{group}')
plt.legend(bbox_to_anchor=(1,1)) # anchor是錨，意思是圖例的框框要放在哪個位置，(0,0)是圖的最左下角，(1,1)是最右上角
plt.show()
