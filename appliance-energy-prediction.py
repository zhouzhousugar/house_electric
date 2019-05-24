# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:26:21 2019

@author: rp.zhou
"""
#导入此次项目需要的package
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.offline as py 
import plotly.graph_objs as go
import os
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
#画图中文正常显示设置
plt.rcParams['font.sans-serif'] =['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")
#读取数据（需要换成自己数据的防放置路径）
data = pd.read_csv(r"D:\sz_electric\analysis_work\kaggle\house_electric\KAG_energydata_complete.csv")
#英文列名改成中文列名
columns = pd.read_csv(r"D:\sz_electric\analysis_work\kaggle\house_electric\columns.csv")
#字段中英文对照变成字典存储
columns_dict={}
for i in range(len(columns)):
    columns_dict[columns.En[i]] = columns.CH[i]
data.rename(columns=columns_dict,inplace=True)
#查看是否替换成功
data.columns  
#描述性统计量
data_d = data.describe()
#删除异常值
data_drop = data[data['整个建筑消耗电量']<=(data['整个建筑消耗电量'].mean()+ data['整个建筑消耗电量'].std()*3)]
#照明消耗电量数据查看
data['照明消耗电量'].value_counts()
#变量分类
col_temp = ['厨房温度', '客厅温度',  '洗衣房温度','办公室温度', '浴室温度',
            '建筑物外温度',  '熨烫室温度','次卧温度', '主卧温度']
col_hum = ['厨房湿度', '客厅湿度', '洗衣房湿度', '办公室湿度',  '浴室湿度',
           '建筑物外侧湿度', '熨烫室湿度',  '次卧湿度', '主卧湿度']
col_weather = ['室外温度', '气压', '室外湿度', '风速','能见度', '露点温度']
col_light = ["照明消耗电量"]
col_target = ["整个建筑消耗电量"]

#查看整个建筑消耗电量分布
data['WEEKDAY'] = ((pd.to_datetime(data['时间']).dt.dayofweek)// 5 == 1).astype(float)
temp_weekday =  data[data['WEEKDAY'] == 0]
temp_weekend =  data[data['WEEKDAY'] == 1]
for i in [data,temp_weekday,temp_weekend]:
    visData = go.Scatter( x= i['时间']  ,  mode = "markers", y = i['整个建筑消耗电量'] )
    layout = go.Layout(yaxis=dict(tickfont=dict(color='rgb(0, 0, 0)',size = 24,)),#设置刻度的字体大小及颜色
                       xaxis=dict(tickfont=dict(color='rgb(0, 0, 0)',size = 24,)))#设置刻度的字体大小及颜色
            
    fig = go.Figure(data=[visData],layout=layout)
    name = u'整个建筑消耗电量分布'+str(len(i))+r'.html'
    py.plot(fig, filename=os.path.join(r'D:\\Code\\zhouzhou',name))
#了解数据形态分布
#整个建筑消耗电量
data[col_target].hist(bins = 50 , figsize= (12,16))
#温度数据
data[col_temp].hist(bins = 50 , figsize= (12,16))
#湿度数据
data[col_hum].hist(bins = 50 , figsize= (12,16))
#天气数据
data[col_weather].hist(bins = 50 , figsize= (12,16))
#核密度曲线
f, ax = plt.subplots(2,2,figsize=(12,8))
vis1 = sns.distplot(data["建筑物外侧湿度"],bins=10, ax= ax[0][0])
vis2 = sns.distplot(data["室外湿度"],bins=10, ax=ax[0][1])
vis3 = sns.distplot(data["能见度"],bins=10, ax=ax[1][0])
vis4 = sns.distplot(data["风速"],bins=10, ax=ax[1][1])

#利用ExtraTrees回归进行共线性检验，剔除变量
data_start = data_drop[col_temp + col_hum + col_weather + col_target]
train, test = train_test_split(data_start,test_size=0.25,random_state=40)
#数据标准化处理
train_standed = pd.DataFrame(StandardScaler().fit_transform(train),
                             columns = train.columns,index=train.index)
test_standed = pd.DataFrame(StandardScaler().fit_transform(test),
                             columns = test.columns,index=test.index)
x_train = train_standed[col_temp + col_hum + col_weather]
y_train = train_standed[col_target]
x_test = test_standed[col_temp + col_hum + col_weather]
y_test = test_standed[col_target]
#ExtraTrees回归模型
etr = ExtraTreesRegressor()
vif_data = pd.Series([variance_inflation_factor(x_train.values.astype(np.float), i) for i in
                         range(x_train.shape[1])],
    index=x_train.columns,name='vif')
#共线性检验并进行剔除
while (vif_data>10).sum()>0:    
    etr.fit(x_train[vif_data.index],y_train)
#得到变量的重要性系数
    selector_data = pd.Series(etr.feature_importances_, index=vif_data.index,name='etr')
    select_etr = np.abs(selector_data).sort_values(ascending=False)
    etr_vif_data = pd.concat([select_etr,vif_data], join='inner',axis=1)
    drop_from_data = etr_vif_data[etr_vif_data['vif']>=10]
    i = len(drop_from_data)-1
    start_columns = list(vif_data.index)
    start_columns.remove(drop_from_data.iloc[i].name)
    vif_data = vif_data[vif_data.index.isin(start_columns)]
#得到模型最终变量：
fina_x_columns = list(vif_data.index)
#删除有问题的变量
fina_x_columns.remove('建筑物外侧湿度')
#查看进入模型的变量情况
x_train[fina_x_columns].head()
#构建线性回归模型
lr = LinearRegression()
lr.fit(x_train[fina_x_columns],y_train)
#得到变量的回归系数
var_coef = pd.Series(lr.coef_[0], index=fina_x_columns)
#在测试集的测试结果
lr.score(x_test[fina_x_columns],y_test)

#计算MAE
y_predict = pd.DataFrame(lr.predict(x_test[fina_x_columns])[0:],columns=['预测值'],index=y_test.index)
mean_absolute_error(y_test,y_predict)
#查看预测值与真实值之间的差距
sns.regplot(x=y_predict['预测值'], y=y_test.iloc[:,0], x_bins=100)



