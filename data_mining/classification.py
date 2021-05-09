import numpy
import pandas as pd
import numpy as np

'''导入数据'''
df = pd.read_csv('./data/credit.csv')
raw = df.copy(deep=True)
cols = df.columns

'''类别数据转化为数值数据'''
unique_columns = ['GENDER', 'MARITAL_STATUS', 'LOANTYPE', 'PAYMENT_TYPE', 'APPLY_TERM_TIME']
# unique_columns = ['GENDER', 'MARITAL', 'LOANTYPE', 'TYPE', 'TIME']
f = open(r'./data/classification_mapping_record.txt', 'w')
for c in unique_columns:
    mapping = {label: idx for idx, label in enumerate(set(df[c]))}
    print(mapping)
    # 把转换的规则保存一下
    f.write(str(mapping) + '\n')
    df[c] = df[c].map(mapping)
f.close()

'''连续数据离散化'''
discrete_columns = ['AGE', 'MONTHLY_INCOME_WHITHOUT_TAX', 'GAGE_TOTLE_PRICE', 'APPLY_AMOUNT', 'APPLY_INTEREST_RATE']
# discrete_columns = ['AGE', 'INCOME', 'PRICE', 'AMOUNT', 'RATE']
discrete_column_index = [2, 5, 7, 8, 10]
df_sub = df[discrete_columns]
des = df_sub.describe()
des.to_csv(r"./data/classify_describe_data_to_discrete.csv", header=True, index=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(des)
pd.reset_option('display.float_format')
# 对于将一个属性划分为几类是一个关键问题
discrete_column_k = [5, 3, 5, 4, 5]
discrete_type = 1


# 可视化
def cluster_plot(cd, ck, raw_data, col_n, way_n, save=True):
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 4))
    for j in range(0, ck):
        plt.plot(raw_data[col_n][cd == j], [j for _i in cd[cd == j]], 'o')
    plt.title(col_n)
    plt.ylim(-0.5, ck - 0.5)
    if save:
        plt.savefig(r'./data/discrete_scatter_' + way_n + '_' + col_n + '.png')
    return plt


for r in range(0, 5):
    i = discrete_column_index[r]
    c = discrete_columns[r]
    k = discrete_column_k[r]
    if discrete_type == 0:
        # 等宽法，存在一个问题，会受到极端值的严重影响
        df.iloc[:, i] = pd.cut(df[c], k, labels=range(k))
        # 画图看效果，果然不是很棒，等频率法不考虑
        cluster_plot(df[c], k, raw, c, way_n='width', save=True)
    elif discrete_type == 1:
        # K-Means聚类法，存在的极端值应该可以被独自划分为一类，类似于将异常值单独考虑的效果
        from sklearn.cluster import KMeans

        data = df[c]
        if r == 1:
            data = data.dropna()
        k_model = KMeans(n_clusters=k).fit(data.values.reshape(-1, 1))
        center = k_model.cluster_centers_
        print(center)
        labels = k_model.labels_
        if r == 1:
            labels = labels.astype(numpy.float64)
            miss_index = np.where(df.isnull())
            for mi in range(0, len(miss_index[0])):
                labels = np.insert(labels, miss_index[0][mi], np.nan)
        df.iloc[:, i] = pd.Series(labels)
        cluster_plot(df[c], k, raw, c, way_n='kmeans', save=True)

'''处理缺失值'''
isNA = df.isnull()
monthly_label = 'MONTHLY_INCOME_WHITHOUT_TAX'
# monthly_label = 'INCOME'
# 经过观察，缺失值只存在于MONTHLY_INCOME_WHITHOUT_TAX
# 缺失值单独归为一类
miss_index = np.where(isNA)
df[monthly_label] = pd.to_numeric(df[monthly_label], errors='coerce')
for i in range(0, len(miss_index[0])):
    df.iloc[miss_index[0][i], miss_index[1][i]] = discrete_column_k[1]
df[monthly_label] = df[monthly_label].astype(int)

'''保存 离散化的规则 离散化后的数据'''


def save_criteria_discrete(f, raw_data, dis_data):
    for sr in range(0, 5):
        sc = discrete_columns[sr]
        f.write(sc + ':\n')
        sk = discrete_column_k[sr]
        for sub_k in range(0, sk + 1):
            sub = raw_data[dis_data[sc] == sub_k]
            if sub.empty:
                continue
            s = str(sub_k) + ': min = ' + str(min(sub[sc])) + ' ,max = ' + str(max(sub[sc]))
            f.write('\t' + s + '\n')


if discrete_type == 0:
    discrete_f = open(r'./data/discrete_criteria_width.txt', 'w')
    save_criteria_discrete(discrete_f, raw, df)
    discrete_f.close()
    df.to_csv(r'./data/discrete_data_width.csv', header=True, index=False)
elif discrete_type == 1:
    discrete_f = open(r'./data/discrete_criteria_kmeans.txt', 'w')
    save_criteria_discrete(discrete_f, raw, df)
    discrete_f.close()
    df.to_csv(r'./data/discrete_data_kmeans.csv', header=True, index=False)

'''决策树'''
import myDecisionTree

data = df.values
# 抛弃第0列，id列
data = data[:, 1:]
# 全部数据训练
labels = list(cols[1:])
# 当前过拟合
mytree = myDecisionTree.create_tree(data, labels, 'c45')
myDecisionTree.create_plot(mytree)
accuracy = myDecisionTree.accuracy(mytree, data, labels)
print(accuracy)
# 剪枝
cut_tree = myDecisionTree.cut_branch_rep(mytree, data, labels)
myDecisionTree.create_plot(cut_tree)
cut_accuracy = myDecisionTree.accuracy(cut_tree, data, labels)
print(cut_accuracy)
prob_count = []
prob_tree = myDecisionTree.count_label(cut_tree, data, labels, '1', prob_count, True)
myDecisionTree.create_plot(prob_tree)
# 剪枝
# 留一交叉验证结果
accuracy_count = []
predict_count = []
for i in range(len(data)):
    print("doing " + str(round(float(100 * (i + 1) / len(data)), 2)) + "%")
    verify_data = data[i]
    left_data = np.delete(data, i, axis=0)
    v_tree = myDecisionTree.create_tree(left_data, labels, 'c45')
    cut_v_tree = myDecisionTree.cut_branch_rep(v_tree, left_data, labels)
    accuracy_count.append(myDecisionTree.accuracy(cut_v_tree, left_data, labels))
    pl = myDecisionTree.predict(cut_v_tree, labels, verify_data)
    if pl == -1:
        predict_count.append(False)
        continue
    if str(pl) == str(verify_data[0]):
        predict_count.append(True)
    else:
        predict_count.append(False)
print(accuracy_count)
print(predict_count)
validation_f = open(r'./data/validation_count.txt', 'w')
for i in range(len(accuracy_count)):
    validation_f.write(str(accuracy_count[i][2]) + ' ' + str(predict_count[i]) + '\n')
validation_f.close()

'''结果评价'''
all_labels = [ex[0] for ex in data]
all_prob = [myDecisionTree.predict(cut_tree, labels, ex) for ex in data]
# ROC
myDecisionTree.eval_plot(all_labels, all_prob, 'ROC', './data/roc.png')
# KS
myDecisionTree.eval_plot(all_labels, all_prob, 'KS', './data/ks.png')
