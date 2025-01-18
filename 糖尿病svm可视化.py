import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from IPython.display import display

import warnings

from sklearn.svm import SVC

warnings.filterwarnings('ignore')

#读取数据
data=pd.read_csv('diabetes_data.csv')
#print(data.head())

#去掉user_id和date
data.drop(['user_id','date'],axis=1,inplace=True)
#print(data.head())

#先对数据集打标签
data['category']=0  #增加一列，表示数据的类别（高中低风险）
#print(data.head())

# 使用条件语句来设置category的值
data.loc[data['risk_score'] < 30, 'category'] = 0
data.loc[(data['risk_score'] >= 30) & (data['risk_score'] <= 60), 'category'] = 1
data.loc[data['risk_score'] > 60, 'category'] = 2

print(data.head())

#pca主成分分析 可视化 聚类  将若干特征投影到二维平面
# Preprocess data: Select numerical columns for clustering and PCA analysis
numerical_cols = ['weight', 'height', 'blood_glucose', 'physical_activity','diet','medication_adherence','stress_level',
                  'sleep_hours', 'hydration_level','bmi' ]

# Standardize the numerical data for clustering and PCA
scaler = StandardScaler()
sd_data = scaler.fit_transform(data[numerical_cols])

# Perform PCA analysis
pca = PCA(n_components=2)
pca_result = pca.fit_transform(sd_data)
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
print(pca_df)

# 查看主成分与原始特征的关系
components = pca.components_
print("PC1与原始特征的关系：")
for i, col in enumerate(numerical_cols):
    print(col, ":", components[0][i])
print("PC2与原始特征的关系：")
for i, col in enumerate(numerical_cols):
    print(col, ":", components[1][i])


X=pca_result
y=data.loc[:,'category']
#creat svm
svm_model=svm.SVC(kernel='rbf',C=1000)
svm_model.fit(X,y)




# 绘制决策边界
def plot_decision_boundary(model, X, y):
    h = .02  # 步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    #plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

    plt.contour(xx, yy, Z,  alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y)

    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.title('svm-rbf')
    plt.show()


# 绘制决策边界
plot_decision_boundary(svm_model, X, y)
