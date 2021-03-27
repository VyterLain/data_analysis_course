import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import seaborn as sns

df = pd.read_csv(r"data/abalone.csv")
print(df)

'''
for all data
'''

# des = df.describe()
# des.to_csv(r"output/description.csv", index=True, header=True)
# print(des)

corr = df.corr()
# corr.to_csv(r"output/corr.csv", index=True, header=True)

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111)
# plt.title("abalone correlation")
# labels = ['Length', 'Diam', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings', ]
# p = sns.heatmap(corr, ax=ax, annot=True, xticklabels=labels, yticklabels=labels, cmap=plt.get_cmap('Greens'))
# plt.savefig(r"output/corr_heatmap.png")
# plt.show()

# sns.set(style='whitegrid', color_codes=True)
# sns.boxplot(x=df['Sex'], y=df['Length'])
# plt.show()

'''
for group data, group by sex
'''

group = df.groupby('Sex')
# for key, df_sex in group:
#     df_sex.to_csv(r"data/description_" + key + ".csv", index=False, header=True)
#     df_sex.describe().to_csv(r"output/description_" + key + ".csv", index=True, header=True)

# for key, df_sex in group:
#     corr = df_sex.corr()
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111)
#     plt.title("abalone correlation of " + key)
#     labels = ['Length', 'Diam', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings', ]
#     p = sns.heatmap(corr, ax=ax, annot=True, xticklabels=labels, yticklabels=labels, cmap=plt.get_cmap('Greens'))
#     plt.savefig(r"output/corr_heatmap_" + key + ".png")
#     plt.show()

# df_F = group.get_group('F')
# df_M = group.get_group('M')
# df_I = group.get_group('I')
