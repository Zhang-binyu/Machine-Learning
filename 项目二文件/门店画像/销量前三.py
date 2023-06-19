import pandas as pd

# 读取CSV文件
df = pd.read_csv('商品货单表.csv', encoding='gbk')

# 按店名进行分组
groups = df.groupby('店名')

# 定义一个空的DataFrame用于保存结果
df_top3 = pd.DataFrame()

# 遍历每个分组
for name, group in groups:
    # 对分组内的数据按照销售数量进行排序
    sorted_group = group.sort_values('个数', ascending=False)
    # 选取销量前三的商品
    top3 = sorted_group.head(3)
    # 将选取的结果添加到df_top3中
    df_top3 = pd.concat([df_top3, top3])

# 将结果保存到新的CSV文件中
df_top3.to_csv('销量前三.csv', index=False)
