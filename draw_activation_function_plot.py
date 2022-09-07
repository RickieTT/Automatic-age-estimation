import itertools

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def tanh(x):
#     return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
#
# def relu(x):
#     return np.maximum(0, x)
#
# def softmax(x):
#     return np.exp(x)/np.sum(np.exp(x), axis=0)
#
#
# def get_central_ax():
#     ax = plt.gca()  # get current axis 获得坐标轴对象
#
#     # 将右边 上边的两条边颜色设置为空 其实就相当于抹掉这两条边
#     ax.spines['right'].set_color('none')
#     ax.spines['top'].set_color('none')
#
#     # 指定下边的边作为 x 轴 指定左边的边为 y 轴
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#
#     ax.spines['bottom'].set_position(('data', 0))  # 指定 data 设置的bottom(也就是指定的x轴)绑定到y轴的0这个点上
#     ax.spines['left'].set_position(('data', 0))
#
#     return ax
#
# x = np.arange(-6.0, 6.0, 0.1)
# # y1 = sigmoid(x)
# # y2 = tanh(x)
# # y3 = relu(x)
# y4 = softmax(x)
#
#
# ax = get_central_ax()
#
# # ax = plt.subplot(111)
# # ax.plot(x, y1)
# # ax.plot(x, y2)
# # ax.plot(x, y3)
# ax.plot(x, y4)
# # ax.plot(x, y4, linestyle='--')
# # ax.legend(['sigmoid'])
# # ax.legend(['tanh'])
# ax.legend(['Softmax'])
# plt.show()

# 显示混淆矩阵
from sklearn.metrics import confusion_matrix


test = pd.read_csv('age_test_1.csv')
pred = pd.read_csv('sub02_1.csv')

len_test = len(test)
# pred_len = len(pred)

test_group = []
pred_group = []
# print(train.head())
# print(test.head())
for index in range(len_test-1):
    test_group.append(test.group[index])
    pred_group.append(pred.group[index])

# print(test_group)
# print(pred_group)


# def plot_confuse(y_test, pred):
#    # 获得预测结果
#    #  predictions = predict(model,x_val)
#     #获得真实标签
#     truelabel = y_test.argmax(axis=-1)   # 将one-hot转化为label
#     cm = confusion_matrix(y_true=truelabel, y_pred=pred)
#     plt.figure()
#     # 指定分类类别
#     classes = range(np.max(truelabel)+1)
#     title='Confusion matrix'
#    #混淆矩阵颜色风格
#     cmap=plt.cm.jet
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     thresh = cm.max() / 2.
#    # 按照行和列填写百分比数据
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()

C = confusion_matrix(test_group, pred_group,labels=["0to6","7to12","13to18","19to25","26to34","35to45","46to60","over60"]) # 可将'1'等替换成自己的类别，如'cat'。
print(test_group)
print(pred_group)
# print(C)

C = C.astype('float') / C.sum(axis=1)[:, np.newaxis]
plt.matshow(C, cmap=plt.cm.Reds) # 根据最下面的图按自己需求更改颜色
# plt.imshow(C,interpolation='nearest')
# for i in range(len(C)):
#     for j in range(len(C)):
for i, j in itertools.product(range(C.shape[0]), range(C.shape[1])):
        plt.text(j, i, '{:.2f}'.format(C[i, j]), horizontalalignment="center")

# plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

plt.ylabel('True label')
plt.xlabel('Predicted label')
# plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
# plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
plt.xticks(range(0,8), labels=["0to6","7to12","13to18","19to25","26to34","35to45","46to60","over60"]) # 将x轴或y轴坐标，刻度 替换为文字/字符
plt.yticks(range(0,8), labels=["0to6","7to12","13to18","19to25","26to34","35to45","46to60","over60"])

plt.tight_layout()
plt.show()
