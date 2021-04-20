import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv(r"data/abalone.csv")
group = df.groupby('Sex')
keys = ['M', 'F', 'I']

'''
for all data, "rings" is dependent, scatter
'''
# fig1 = plt.figure(figsize=(12, 16))
# for i in range(1, 8):
#     ax = fig1.add_subplot(3, 3, i)
#     sns.scatterplot(x=df.iloc[:, i], y=df.iloc[:, -1], hue=df.iloc[:, 0], alpha=.9, s=7, ax=ax)
# plt.tight_layout(pad=6, w_pad=2, h_pad=6)
# plt.savefig(r'./output/description/scatter_rings.png')

'''
for group data, group by sex, "rings" is dependent, scatter
'''
# for key, dff in group:
#     fig2 = plt.figure(figsize=(12, 16))
#     for i in range(1, 8):
#         ax = fig2.add_subplot(3, 3, i)
#         sns.scatterplot(x=dff.iloc[:, i], y=dff.iloc[:, -1], alpha=0.9, s=7, ax=ax)
#     plt.tight_layout(pad=6, w_pad=2, h_pad=6)
#     plt.savefig(r'./output/description/scatter_rings_' + key + '.png')

'''
for all data, "sex" is dependent, scatter, box, hist
'''
# fig3_1 = plt.figure(figsize=(12, 16))
# ax3_1 = fig3_1.add_subplot(111)
# sns.scatterplot(x=df.iloc[:, 1], y=df.iloc[:, 2], hue=df.iloc[:, 0], alpha=1, s=7, ax=ax3_1)
# ax3_1.set_title('Length & Diam')
# plt.savefig(r'./output/description/scatter_length_diam.png')

# fig3_2 = plt.figure(figsize=(18, 8))
# for i in range(3):
#     ax3_2 = fig3_2.add_subplot(1, 3, i+1)
#     g = group.get_group(keys[i])
#     sns.scatterplot(x=g.iloc[:, 1], y=g.iloc[:, 2], alpha=1, s=7, ax=ax3_2)
#     ax3_2.set_title(keys[i])
# plt.tight_layout(pad=6, w_pad=6)
# plt.savefig(r'./output/description/scatter_length_diam_sex.png')

# fig4 = plt.figure(figsize=(12, 12))
# for i in range(1, 9):
#     ax4 = fig4.add_subplot(3, 3, i)
#     sns.boxplot(x=df.iloc[:, 0], y=df.iloc[:, i], linewidth=0.5, width=0.6, fliersize=0.4)
#     ax4.set_xlabel('')
# plt.tight_layout(pad=6, w_pad=4, h_pad=3)
# plt.savefig(r'./output/description/box.png')

# for key in keys:
#     fig5 = plt.figure(figsize=(15, 15))
#     g = group.get_group(key)
#     for i in range(1, 9):
#         ax5 = fig5.add_subplot(3, 3, i)
#         sns.histplot(g.iloc[:, i], kde=True, ax=ax5)
#         ax5.set_ylabel('')
#     plt.tight_layout(pad=6, w_pad=3, h_pad=6)
#     plt.savefig(r'./output/description/hist_' + key + '.png')

plt.rcParams["font.size"] = 6
plt.rcParams["axes.labelsize"] = 6
g = sns.PairGrid(df)
g.map(sns.scatterplot, s=0.8)
g.tight_layout(pad=6)
g.savefig(r'./output/description/linear_corr_all.png')
