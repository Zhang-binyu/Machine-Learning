import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 导入商品货单表.csv
df = pd.read_csv('商品货单表.csv', encoding='gbk', parse_dates=['时间'])

# 选择要预测销售额的商品
product_name = '蜀大侠牛油火锅底料1.75kg*10(大红锅流通版)'
product_df = df[df['产品'] == product_name]

# 创建时间序列
time_series = pd.Series(product_df['个数'].values, index=product_df['时间'])

# 选择历史数据区间
start_date = '2022-05-01'
end_date = '2022-07-31'
historical_data = time_series[start_date:end_date]

# 训练ARIMA模型并进行预测
model = ARIMA(historical_data, order=(1, 1, 1))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

# 将预测结果拼接到历史销售额数据上，形成完整的时间序列
full_time_series = time_series.copy()
full_time_series[start_date:end_date] = historical_data
full_time_series[start_date:].iloc[:30] = forecast

# 绘制历史销售额和预测销售额图
plt.figure(figsize=(12, 6))
plt.bar(full_time_series.index, full_time_series, label='销售额')
plt.title(f'{product_name}销售额历史数据与预测结果')
plt.xlabel('日期')
plt.ylabel('销售额')
plt.xticks(rotation=45)
plt.legend()
plt.show()

# 将预测销售额的下标改为日期
forecast.index = pd.date_range(start=end_date, periods=30)

# 绘制仅预测销售额图
plt.figure(figsize=(12, 6))
plt.bar(forecast.index, forecast, label='销售额')
plt.title(f'{product_name}销售额预测结果')
plt.xlabel('日期')
plt.ylabel('销售额')
plt.xticks(rotation=45)
plt.legend()
plt.show()
