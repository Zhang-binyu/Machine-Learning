import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics


# 读取数据
train = pd.read_csv('heart_train.csv')
test = pd.read_csv('heart_test.csv')


# 数据预处理
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


# 简单预处理
train_list = []

for items in train.values:
    train_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])

train = pd.DataFrame(np.array(train_list))
train.columns = ['id'] + ['s_' + str(i) for i in range(len(train_list[0]) - 2)] + ['label']
train = reduce_mem_usage(train)

test_list = []
for items in test.values:
    test_list.append([items[0]] + [float(i) for i in items[1].split(',')] + [items[2]])

test = pd.DataFrame(np.array(test_list))
test.columns = ['id'] + ['s_' + str(i) for i in range(len(test_list[0]) - 2)] + ['label']
test = reduce_mem_usage(test)

train.head()
print(train.head())

test.head()
print(test.head())

# 训练数据/测试数据准备
x_train = train.drop(['id', 'label'], axis=1)
y_train = train['label']
x_test = test.drop(['id', 'label'], axis=1)
y_test = test['label']

print(x_test)


def decide_tree(x_train, y_train, x_test, y_test):
    from sklearn import tree
    dt = tree.DecisionTreeClassifier()

    # 使用训练集数据进行训练
    dt = dt.fit(x_train, y_train)
    # 使用训练好的模型对测试集进行测试，并输出正确率
    train_score = dt.score(x_train, y_train)
    print(y_test.shape)
    print(x_test.shape)
    test_score = dt.score(x_test, y_test)
    print("Train 正确率：", train_score)
    print("Test 正确率：", test_score)

    y_pred = dt.predict(x_test)

    # 召回率
    print("召回率：", metrics.recall_score(y_test, y_pred, average="micro"))
    # 查准率
    print("查准率：", metrics.precision_score(y_test, y_pred, average="micro"))

    from sklearn.metrics import confusion_matrix

    # 混淆矩阵
    confusion = confusion_matrix(y_test, y_pred)
    sns.heatmap(confusion, cmap="YlGnBu_r", fmt="d", annot=True)

    # 热度图
    plt.show()


    # 召回率
    # recall = metrics.recall_score(y_test, y_pred, average='macro')
    # 查准率
    # precision = metrics.precision_score(y_test, y_pred, average='macro')


decide_tree(x_train, y_train, x_test, y_test)