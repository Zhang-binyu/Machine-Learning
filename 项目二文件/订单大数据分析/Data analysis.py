import pandas as pd
import numpy as np

# 读取商品货单表
df = pd.read_excel('商品货单表.xls')

# 按照时间进行排序
df.sort_values('时间', inplace=True)

# 按照店名和产品进行分组
groups = df.groupby(['店名', '产品'])

# 遍历每个分组，计算订货量和提醒时间
results = []
for name, group in groups:
    # 获取当前分组的历史数据
    data = group['个数'].values

    # 计算平均销量和标准差
    mean = np.mean(data)
    std = np.std(data)

    # 计算提醒时间和订货量
    lead_time = pd.Timedelta(days=7)  # 提前7天提醒
    reorder_point = mean + std  # 订货点为平均销量加上标准差
    reorder_quantity = reorder_point - np.max(data)  # 订货量为订货点减去当前库存

    # 如果订货量小于0，则不需要订货
    if reorder_quantity < 0:
        continue

    # 计算提醒时间
    last_order_date = pd.to_datetime(group['时间'].max())  # 将时间转换为时间类型
    remind_date = last_order_date + lead_time  # 提醒时间

    # 将结果保存到列表中
    results.append({
        '店名': name[0],
        '产品': name[1],
        '订货量': reorder_quantity,
        '订货预测时间': remind_date.strftime('%Y-%m-%d')  # 将时间转换为字符串
    })

# 将结果转换为 DataFrame，并导出到 Excel 文件
result_df = pd.DataFrame(results)
result_df.to_excel('订货提醒表.xlsx', index=False)
