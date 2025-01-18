import pandas as pd
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
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
#print(X_train,y_train)

#creat svm
svm_model=svm.SVC(kernel='rbf',C=100000)
svm_model.fit(X_train,y_train)
y_predict=svm_model.predict(X_test)
accuracy=accuracy_score(y_predict,y_test)
print(accuracy)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score



#交叉验证选c值
# 定义SVM模型
svm = SVC(kernel='rbf')

# 定义C的候选值列表
param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000, 100000]}

# 创建GridSearchCV对象，使用5折交叉验证
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=10, scoring=make_scorer(accuracy_score))

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最佳C值
print("Best C:", grid_search.best_params_['C'])

# 使用最佳参数的模型
best_svm = grid_search.best_estimator_

# 在测试集上评估模型性能
y_pred = best_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




