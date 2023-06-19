import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import json
import datetime
from mlxtend.frequent_patterns import apriori, association_rules

path_ordelModel = r"OrderModel订单表.csv"

df = pd.read_csv(path_ordelModel)

# 选择所需列并筛选数据
df2 = df.loc[df['CustomName'] == '青羊区周记蜀大侠火锅店(西玉龙二店直营店）', ['CreateTime', 'Goods']].copy()
# 将 Goods 列中的 JSON 字符串解析成列表，并将列表中的字典转换成 DataFrame
df2 = df2.dropna(how='any')
goods_list = df2['Goods'].apply(lambda x: [{'CreateTime': d['CreateTime'][:10], 'GoodsName': d['GoodsName']} for d in json.loads(str(x))])
goods_df = pd.DataFrame([item for sublist in goods_list for item in sublist])

# 计算每个商品的购买次数，生成透视表
df2 = pd.pivot_table(goods_df.assign(times=1), index='CreateTime', columns='GoodsName', values='times', aggfunc='sum', fill_value=0)
df2 = df2.astype('bool')

print(df2)

frequent_itemsets = apriori(df2, min_support=0.2, use_colnames=True)
frequent_itemsets.sort_values('support',inplace=True, ascending=False)
print(frequent_itemsets)