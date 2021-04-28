import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv('./data/credit.csv')

'''类别数据转化为数值数据'''
# unique_columns = ['GENDER', 'MARITAL_STATUS', 'LOANTYPE', 'PAYMENT_TYPE', 'APPLY_TERM_TIME']
# f = open('classification_mapping_record.txt', 'w')
# for c in unique_columns:
#     mapping = {label: idx for idx, label in enumerate(set(df[c]))}
#     # 把转换的规则保存一下
#     f.write(str(mapping) + '\n')
#     df[c] = df[c].map(mapping)
# f.close()

'''连续数据离散化'''
discrete_columns = ['AGE', 'MONTHLY_INCOME_WHITHOUT_TAX', 'GAGE_TOTLE_PRICE', 'APPLY_AMOUNT', 'APPLY_INTEREST_RATE']
df_sub = df[discrete_columns]
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
print(df_sub.describe())
pd.reset_option('display.float_format')
# 等宽法
# 等频率法
# K-Means聚类法
