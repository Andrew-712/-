import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from IPython.display import display

import warnings
warnings.filterwarnings('ignore')
#读取数据
data=pd.read_csv('diabetes_data.csv')
#print(data.head())

#去掉user_id和date
data.drop(['user_id','date'],axis=1,inplace=True)
#print(data.head())

#下面用决策树分类
#先对数据集打标签
data['category']=0  #增加一列，表示数据的类别（高中低风险）
#print(data.head())

# 使用条件语句来设置category的值
data.loc[data['risk_score'] < 30, 'category'] = 0
data.loc[(data['risk_score'] >= 30) & (data['risk_score'] <= 60), 'category'] = 1
data.loc[data['risk_score'] > 60, 'category'] = 2

print(data.head())



X=data.drop(['category','risk_score'],axis=1)
y=data.loc[:,'category']
print(X.shape,y.shape)

#split it into train set and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print(X_test)


#决策树
from sklearn import tree
dc_tree=tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=5,max_depth=3,min_samples_split=20)
dc_tree.fit(X_train,y_train)


y_predict=dc_tree.predict(X_test)
print(y_predict)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)

#决策树可视化
import matplotlib as mpl
font2={'family':'SimHei',
       'weight':'normal',
       'size':200,}
mpl.rcParams['font.family']='SimHei'
mpl.rcParams['axes.unicode_minus']=False
#matplotlib inline
from matplotlib import pyplot as plt
fig=plt.figure(figsize=(200,200))
tree.plot_tree(dc_tree,filled=True,
               feature_names=['weight','height','blood_glucose','physical_activity','diet','medication','stress','sleep','hydration','bmi'],
               class_names=['low_risk','moderate_risk','high_risk'])
plt.savefig('C:/Users/19816/Desktop/人工智能大作业/ai 大作业/decision_tree_display.png',bbox_inches='tight',pad_inches=0.0)
