import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.nn import MSELoss
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('diabetes_data.csv')

# 去掉 user_id 和 date
data.drop(['user_id', 'date'], axis=1, inplace=True)

# 打标签
data['category'] = 0
data.loc[data['risk_score'] < 30, 'category'] = 0
data.loc[(data['risk_score'] >= 30) & (data['risk_score'] <= 60), 'category'] = 1
data.loc[data['risk_score'] > 60, 'category'] = 2

X = data.drop(['category', 'risk_score'], axis=1)
y = data['category']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 tensor
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)
print(y_train.shape)
#y_train = torch.argmax(y_train, dim=1)
#y_test = torch.argmax(y_test, dim=1)

# 定义模型
input_size = X_train.shape[1]
print(input_size)
hidden1_size = 58
output_size = 3

model = nn.Sequential(
    nn.Linear(input_size, hidden1_size),
    nn.ReLU(),
    nn.Linear(hidden1_size, output_size)
)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
#交叉熵对比了模型的预测结果和数据的真实标签，随着预测越来越准确，交叉熵的值越来越小，如果预测完全正确，交叉熵的值就为0。
# 因此，训练分类模型时，可以使用交叉熵作为损失函数。
optimizer = Adam(model.parameters(), lr=0.008)

# 训练模型
num_epochs = 1001
losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    #loss = criterion(outputs, y_train.unsqueeze(1).float())
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# 绘制损失曲线
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

'''
y_predict=model(X_test)
print(y_predict)
y_test=y_test.detach().numpy()
y_predict=y_predict.detach().numpy()
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)
print(accuracy)
'''


# 获取测试集上的预测结果并转换为类别索引形式
y_predict = model(X_test)
_, predicted_labels = torch.max(y_predict, dim=1)  # 获取每行概率最大的类别索引
y_predict = predicted_labels.detach().numpy()  # 转换为numpy数组

y_test = y_test.detach().numpy()

# 计算准确率
accuracy = accuracy_score(y_test, y_predict)
print("accuracy=" ,accuracy)



