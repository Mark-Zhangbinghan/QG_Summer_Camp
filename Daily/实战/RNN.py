import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras import models
from tensorflow.keras.layers import Dense, SimpleRNN

"""
model = Sequential()
model.add(SimpleRNN(units=5,
                    input_shape=(X.shape[1], X.shape[2]),
                    activation='relu'))         # activation-激活函数
model.add(Dense(units=1, actvation='linear'))  # units-预测几个数据
model.compile(optimizer='adam', loss='mean_squared_error')  # 用最小损失误差做损失函数
"""

data = pd.read_csv('zgpa_train.csv')
price = data.loc[:, 'close']
price_norm = price / price.max()    # 归一化处理
time_step = 8                       # 时间戳（步长）

# 归一化之前的市场价
fig1 = plt.figure(figsize=(8, 5))
plt.plot(price)
plt.title('close price')
plt.xlabel('time')
plt.ylabel('price')
plt.show()


# 对x和y赋值
def extract_data(data, time_step):  # 希望通过前time_step个数据对第time_step+1个数据做预测
    x = []
    y = []
    for i in range(len(data) - time_step):
        x.append(data[i:i+time_step])
        y.append(data[i+time_step])
    x = np.array(x)
    x = x.reshape(x.shape[0], x.shape[1], 1)
    return x, y


X, y = extract_data(price_norm, time_step)

model = models.Sequential()
model.add(SimpleRNN(units=5, input_shape=(time_step, 1), activation='relu'))  # 输入层
model.add(Dense(units=1, activation='linear'))  # 输出层
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X, y, batch_size=30, epochs=200)
y_pred = model.predict(X)*max(price)
y_train = [i*max(price) for i in y]

# 检验模型
data_test = pd.read_csv('zgpa_test.csv')
price_test = data_test.loc[:, 'close']
price_test_norm = price_test / price.max()
x_test_norm, y_test_norm = extract_data(price_test_norm, time_step)
y_pred_test = model.predict(x_test_norm)*max(price)
y_test = [i*max(price) for i in y_test_norm]


