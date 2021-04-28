import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv('./data/credit.csv')
# pd.set_option('display.max_columns', None)
cols = df.columns

'''类别数据转化为数值数据'''
unique_columns = ['GENDER', 'MARITAL_STATUS', 'LOANTYPE', 'PAYMENT_TYPE', 'APPLY_TERM_TIME']
# f = open(r'./data/classification_mapping_record.txt', 'w')
for c in unique_columns:
    mapping = {label: idx for idx, label in enumerate(set(df[c]))}
    print(mapping)
    #     # 把转换的规则保存一下
    #     f.write(str(mapping) + '\n')
    df[c] = df[c].map(mapping)
# f.close()

'''连续数据离散化'''
discrete_columns = ['AGE', 'MONTHLY_INCOME_WHITHOUT_TAX', 'GAGE_TOTLE_PRICE', 'APPLY_AMOUNT', 'APPLY_INTEREST_RATE']
discrete_column_index = [2, 5, 7, 8, 10]
df_sub = df[discrete_columns]
des = df_sub.describe()
# des.to_csv(r"./data/classify_describe_data_to_discrete.csv", header=True, index=True)
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(des)
# pd.reset_option('display.float_format')
# 对于将一个属性划分为几类是一个关键问题
discrete_column_k = [5, 5, 5, 5, 5]
discrete_type = 2
for r in range(0, 5):
    i = discrete_column_index[r]
    c = discrete_columns[r]
    k = discrete_column_k[r]
    if discrete_type == 0:
        # 等宽法，存在一个问题，会受到极端值的严重影响
        df.iloc[:, i] = pd.cut(df[c], k, labels=range(k))
    elif discrete_type == 1:
        # 等频率法，存在一个问题，可能会将不该是一类的数据分在一起
        df.iloc[:, i] = pd.qcut(df[c], k, labels=range(k))
    elif discrete_type == 2:
        # K-Means聚类法
        useless = 1  # TODO

'''处理缺失值'''
isNA = df.isnull()
print(isNA.any())
print()
# 经过观察，缺失值只存在于MONTHLY_INCOME_WHITHOUT_TAX
# 缺失值单独归为一类
miss_index = np.where(isNA)
df['MONTHLY_INCOME_WHITHOUT_TAX'] = pd.to_numeric(df['MONTHLY_INCOME_WHITHOUT_TAX'], errors='coerce')
for i in range(0, len(miss_index[0])):
    df.iloc[miss_index[0][i],miss_index[1][i]] = discrete_column_k[1]
df['MONTHLY_INCOME_WHITHOUT_TAX'] = df['MONTHLY_INCOME_WHITHOUT_TAX'].astype(int)
# print(max(df['MONTHLY_INCOME_WHITHOUT_TAX']))
# if discrete_type == 0:
#     df.to_csv(r'./data/discrete_data_width.csv', header=True, index=True)
# elif discrete_type == 1:
#     df.to_csv(r'./data/discrete_data_freq.csv', header=True, index=True)
# elif discrete_type == 2:
#     df.to_csv(r'./data/discrete_data_kmeans.csv', header=True, index=True)
