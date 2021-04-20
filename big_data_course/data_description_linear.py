import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv(r"data/abalone.csv")

'''
for all data, "rings" is dependent, scatter
'''
fig1 = plt.figure(figsize=(12, 16))
for i in range(1, 8):
    ax = fig1.add_subplot(3, 3, i)
    sns.scatterplot(x=df.iloc[:, i], y=df.iloc[:, -1], hue=df.iloc[:, 0], alpha=.9, s=7, ax=ax)
plt.tight_layout(pad=6, w_pad=2, h_pad=6)
plt.savefig(r'./output/description/scatter_rings.png')

'''
for group data, group by sex, "rings" is dependent, scatter
'''
group = df.groupby('Sex')

'''
for all data, "sex" is dependent, scatter, box, facetGrid
'''
# fig3 = plt.figure()
# ax3 = fig3.add_subplot(111)
# sns.scatterplot(x=df.iloc[:, 1], y=df.iloc[:, 2], hue=df.iloc[:, 0], alpha=0.9, s=5, ax=ax3)
# ax3.set_title('Length & Diam')
# plt.savefig(r'./output/description/scatter_for_sex.png')
# plt.show()
