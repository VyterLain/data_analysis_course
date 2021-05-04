import math
import operator
import random
import copy

import numpy as np
from matplotlib import pyplot as plt

'''
两个决策树：ID3 和 C4.5
该包中的功能
part 1 手写
1、创建决策树 => done: create_tree(train, labels: list, tree_type='c45')
2、进行剪枝（策略：
3、功能函数 => todo: predict(tree, labels, obj), accuracy(tree, data, labels: list)
part 2 调包
1、使用sklearn中的决策树算法进行训练
part 3 ROC 和 KS => done: eval_plot(true_label: list, prob: list, eval_type='ROC', save_path=None)
1、ROC曲线 => done
2、KS曲线 => done
part 4 画决策树 => done: create_plot(tree)
'''

"""
写在最前面，关于代码中的数据集，默认以classification.py中处理的数据集格式为参考，即
1、使用前将dataframe转为numpy矩阵
3、第1列为label，index=0
4、其他列为feature
"""


def calculate_entropy(data):
    """
    计算信息熵 entropy = - sum(P*log2(P))
    :param data:数据data，为numpy矩阵，可以对dataframe使用.values转换，其中标签在index=0
    :return:熵
    """
    label_count = {}
    for item in data:
        label = item[0]
        label_count[label] = label_count.get(label, 0) + 1
    entropy = 0.0
    for key in label_count:
        p = float(label_count[key]) / len(data)
        entropy = entropy - p * math.log(p, 2)
    return entropy


def filter_by_feature(data, index, feature):
    """
    根据提供的特征值，获得该特征值对应的记录（其中不包括该特征值）
    用于树中一个节点往下分叉的数据的提取
    :param data:数据data
    :param index: 该特征的索引
    :param feature:特征值
    :return:筛选后的结果
    """
    res = []
    for item in data:
        if item[index] == feature:
            tmp = []
            tmp.extend(item[:index])
            tmp.extend(item[index + 1:])
            res.append(tmp)  # 得到特征值对应的除了index索引属性的记录内容
    res = np.array(res)
    return res


def count_majority_labels(class_list: list):
    """
    labels中数量最多的那个label
    在对最后的特征进行分类的时候，返回占多数的那个label
    :param class_list: 全部labels
    :return: 占多数的label值
    """
    class_count = {}
    for c in class_list:
        class_count[c] = class_count.get(c, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def choose_best_feature_id3(train):
    """
    找到当前数据集的最佳id3方法下的最佳划分特征
    信息增益：Gain(D,a) = Entropy(D) - sum((|a| / |D|) * Entropy(a))
    :param train: 数据train
    :return: 最佳划分的索引
    """
    entropy = calculate_entropy(train)
    gain = 0.0
    best_feature = -1
    for i in range(1, len(train[0])):
        feature_set = set(ex[i] for ex in train)
        new_entropy = 0.0
        for feature in feature_set:
            filter_data = filter_by_feature(train, i, feature)
            prob = float(len(filter_data)) / len(train)
            new_entropy = new_entropy + prob * calculate_entropy(filter_data)
        new_gain = entropy - new_entropy
        if new_gain > gain:
            gain = new_gain
            best_feature = i
    return best_feature


def choose_best_feature_c45(train):
    """
    找到当前数据集的最佳c4.5方法下的最佳划分特征
    信息增益：IV(a) = -sum((|a| / |D|) * log2(|a| / |D|)); GainRate(D,a) = Gain(D,a) / IV(a)
    :param train: 数据train
    :return: 最佳划分的索引
    """
    entropy = calculate_entropy(train)
    gain_rate = 0.0
    best_feature = -1
    for i in range(1, len(train[0])):
        feature_set = set(ex[i] for ex in train)
        new_entropy = 0.0
        iv = 0.0
        for feature in feature_set:
            filter_data = filter_by_feature(train, i, feature)
            prob = float(len(filter_data)) / len(train)
            new_entropy = new_entropy + prob * calculate_entropy(filter_data)
            tmp = 0.0
            if prob != 0:
                tmp = math.log(prob, 2)
            iv = iv - prob * tmp
        # 当iv值为0时，说明prob为0，即当前特征的特征值下没有数据
        # 因此，信息增益也一定为0，所以不用考虑iv的值了，避免出现divided_by_zero的异常
        if iv == 0:
            new_gain = 0.0
        else:
            new_gain = (entropy - new_entropy) / iv
        if new_gain > gain_rate:
            gain_rate = new_gain
            best_feature = i
    return best_feature


def create_tree(train, labels: list, tree_type='c45'):
    """
    使用递归的方式构建决策树
    (标签=>值=>类型) => (标签=>值=>类型) => ...
    :param train: 训练数据
    :param labels: 标签
    :param tree_type:构建树的方法
    :return:树
    """
    _labels = labels[:]
    label_list = [ex[0] for ex in train]
    # 递归会在当前分类全部label都相同的时候停止
    # count方法属于string，因此将list转换成string之后在使用该方法
    if str(label_list).count(str(label_list[0])) == len(label_list):
        return label_list[0]
    # 当划分的数据集只有一列数据时，即label，说明当前已经没有可以划分的属性了，递归结束
    if len(train[0]) == 1:
        return count_majority_labels(label_list)
    best_feature = -1
    if tree_type == 'id3':
        best_feature = choose_best_feature_id3(train)
    elif tree_type == 'c45':
        best_feature = choose_best_feature_c45(train)
    if best_feature == -1:
        # 这说明存在条件相同的两个记录，但是label不同，随机选择一个feature
        best_feature = random.randint(1, len(_labels) - 1)
    best_feature_label = _labels[best_feature]
    tree = {best_feature_label: {}}
    feature_set = set([ex[best_feature] for ex in train])
    # 递归构建分类树
    del _labels[best_feature]
    for feature in feature_set:
        next_train = filter_by_feature(train, best_feature, feature)
        tree[best_feature_label][feature] = create_tree(next_train, _labels, tree_type)
    return tree


def predict(tree, labels, obj):
    """
    给定一个数据，根据决策树预测类型
    :param tree: 决策树模型
    :param labels: 标签列表
    :param obj: 待预测数据
    :return: 分类
    """
    root = list(tree.keys())[0]
    next_node = tree[root]
    index = labels.index(root)  # 找到root在标签中所对应的索引
    key = obj[index]
    val = next_node[key]
    # 判断是否到达叶节点
    if isinstance(val, dict):
        return predict(val, labels, obj)
    else:
        return val


def count_accuracy(tree, data, labels: list, count: list):
    """
    计算叶子节点的预测准确度
    :param tree: 树
    :param data: 测试数据
    :param labels: 标签列表
    :param count: 记录结果用的列表
    :return: 结果存在count列表中
    """
    _labels = labels[:]
    root = list(tree.keys())[0]
    next_node = tree[root]
    index = _labels.index(root)
    del _labels[index]
    for feature in next_node.keys():
        true_count = 0
        false_count = 0
        filter_data = filter_by_feature(data, index, feature)
        if isinstance(next_node[feature], dict):
            count_accuracy(next_node[feature], filter_data, _labels, count)
        else:
            for d in filter_data:
                # 防止前面在处理中出现的问题，统一转换为string对比
                if str(d[0]) == str(next_node[feature]):
                    true_count += 1
                else:
                    false_count += 1
            count.append([true_count, false_count])


def count_label(tree, data, labels: list, label: str, count: list, generate_prob_tree=False):
    """
    计算叶子节点的中两种label的数量
    :param generate_prob_tree:是否生成概率树，是则有返回值
    :param label: 要计算的label
    :param tree: 树
    :param data: 测试数据
    :param labels: 标签列表
    :param count: 记录结果用的列表
    :return: 结果存在count列表中
    """
    if generate_prob_tree:
        if not isinstance(tree, dict):
            return float(count[-1][0] / (count[-1][0] + count[-1][1]))
    _labels = labels[:]
    root = list(tree.keys())[0]
    p_tree = {root: {}}
    next_node = tree[root]
    index = _labels.index(root)
    del _labels[index]
    for feature in next_node.keys():
        label_count = 0
        other_count = 0
        filter_data = filter_by_feature(data, index, feature)
        if isinstance(next_node[feature], dict):
            if generate_prob_tree:
                p_tree[root][feature] = count_label(next_node[feature], filter_data, _labels, label, count,
                                                    generate_prob_tree)
            else:
                count_label(next_node[feature], filter_data, _labels, label, count)
        else:
            for d in filter_data:
                # 防止前面在处理中出现的问题，统一转换为string对比
                if str(d[0]) == label:
                    label_count += 1
                else:
                    other_count += 1
            count.append([label_count, other_count])
            if generate_prob_tree:
                p_tree[root][feature] = count_label(next_node[feature], filter_data, _labels, label, count,
                                                    generate_prob_tree)
    if generate_prob_tree:
        return p_tree


"""
def cutBranch(tree, data, lables):
    '''
    悲观剪枝 参考：http://www.jianshu.com/p/794d08199e5e
    old = errorNum + o.5 * L  errorNum:叶子节点错误分类的个数，L：叶子节点个数
    p = old / N  N:数据样本总个数
    new = errorNum + o.5
    S = math.sqrt(N  * p * (1 - p))
    if new <= old - S then 剪枝

    注：都是自己理解的，如果有不对的地方欢迎指出
    '''
    root = list(tree.keys())[0]
    nextNode = tree[root]
    index = lables.index(root)
    newTree = {root: {}}
    del(lables[index])
    for key in nextNode.keys():
        # 如果子节点不是叶子节点就判断其是否满足剪枝
        if(isinstance(nextNode[key], dict)):
            A = getLablesByfeature(data, index, key)
            count = []
            # 获取每个叶子节点的(正确分类数，错误分类数)
            getCount(nextNode[key], A, lables[:], count)
            allnum = 0
            errornum = 0
            for i in count:
                allnum += i[0] + i[1]
                errornum += i[1]
            if(errornum == 0):
                # 当该子树不存在错误分类的时候，不对该树进行剪枝操作
                # 进行下个循环
                newTree[root][key] = nextNode[key]
                continue
            old = errornum + len(count) * 0.5
            new = errornum + 0.5
            p = old / allnum
            S = math.sqrt(allnum * p * (1 - p))
            if(new <= old - S):
                # 用当前分类时出现最多的lables代替该子树
                classList = [item[-1] for item in A]
                newTree[root][key] = majorityCnt(classList)
            else:
                # 不满足剪枝则进入其子树内部继续进行剪枝操作
                newTree[root][key] = cutBranch(nextNode[key], A, lables[:])
        else:
            newTree[root][key] = nextNode[key]
    return newTree
"""


def accuracy(tree, data, labels: list):
    """
    计算所有数据中，正确分类的数量、错误分类的数量，正确率
    :param tree: 树
    :param data: 原始数据
    :param labels: 标签列表
    :return: 正确分类数量、错误分类数量、正确率
    """
    count = []
    count_accuracy(tree, data, labels, count)
    true_num = 0
    false_num = 0
    for d in count:
        true_num += d[0]
        false_num += d[1]
    return true_num, false_num, float(true_num / (true_num + false_num))


decision = dict(boxstyle="round4", color='#3366FF')  # 决策结点形态
leaf = dict(boxstyle="circle", color='#FF6633')  # 叶结点形态
arrow = dict(arrowstyle="<-", color='g')  # 连接线


def create_plot(tree):
    """
    绘制完整的图
    :param tree:树
    """
    fig = plt.figure()
    fig.clf()
    # 清除坐标轴
    ax_props = dict(xticks=[], yticks=[])
    create_plot.ax = plt.subplot(111, frameon=False, **ax_props)
    # 初始化plot_tree中的参数
    plot_tree.w = float(get_leaf_num(tree))
    plot_tree.d = float(get_tree_depth(tree))
    # 横坐标偏移量，向父节点位置向左进行相对的偏移
    plot_tree.x = -0.5 / plot_tree.w
    plot_tree.y = 1.0
    # 父节点放在(0.5, 1)的坐标位置
    plot_tree(tree, (0.5, 1.0), '')
    plt.show()


def plot_node(text, son, parent, node_type):
    """
    plt.annotate(str, xy=data_point_position, xytext=annotate_position,
             va="center",  ha="center", xycoords="axes fraction",
             textcoords="axes fraction", bbox=annotate_box_type, arrowprops=arrow_style)
    str是给数据点添加注释的内容，支持输入一个字符串
    xy=是要添加注释的数据点的位置
    xytext=是注释内容的位置
    bbox=是注释框的风格和颜色深度，fc越小，注释框的颜色越深，支持输入一个字典
    va="center",  ha="center"表示注释的坐标以注释框的正中心为准，而不是注释框的左下角(v代表垂直方向，h代表水平方向)
    xycoords和textcoords可以指定数据点的坐标系和注释内容的坐标系，通常只需指定xycoords即可，textcoords默认和xycoords相同
    arrowprops可以指定箭头的风格支持，输入一个字典
    plt.annotate()的详细参数可用__doc__查看，如：print(plt.annotate.__doc__)
    """
    create_plot.ax.annotate(text, xy=parent, xytext=son, bbox=node_type, arrowprops=arrow)


def plot_text_mid(son, parent, text):
    x_mid = (parent[0] - son[0]) / 2.0 + son[0]
    y_mid = (parent[1] - son[1]) / 2.0 + son[1]
    create_plot.ax.text(x_mid, y_mid, text, rotation=30)


def plot_tree(tree, parent, text):
    """
    递归绘制树
    :param tree:树
    :param parent:父节点坐标
    :param text: 父节点与子节点之间的文字信息
    """
    leaf_num = get_leaf_num(tree)
    label = list(tree.keys())[0]
    # 子节点的坐标，考虑该节点下所有的叶节点数量对x向右进行偏移，这个1.0可以保证在访问到叶节点时候，将初始偏移量影响取消
    son = (plot_tree.x + (1.0 + float(leaf_num)) / 2.0 / plot_tree.w, plot_tree.y)
    # 绘制父子节点之间的内容
    plot_text_mid(son, parent, text)
    # 绘制带箭头的节点
    plot_node(label, son, parent, decision)
    next_node = tree[label]
    # 向下推进，进入到下一个节点的y
    plot_tree.y = plot_tree.y - 1.0 / plot_tree.d
    for feature in next_node.keys():
        if isinstance(next_node[feature], dict):
            plot_tree(next_node[feature], son, str(feature))
        else:
            # 完成一个叶节点，向右进行偏移，到达下一个叶节点的x
            plot_tree.x = plot_tree.x + 1.0 / plot_tree.w
            # 画叶节点
            plot_node(next_node[feature], (plot_tree.x, plot_tree.y), son, leaf)
            # 补充叶节点和父节点间的特征值文本信息
            plot_text_mid((plot_tree.x, plot_tree.y), son, str(feature))
    # 回退到上一个节点的y
    plot_tree.y = plot_tree.y + 1.0 / plot_tree.d


def get_leaf_num(tree):
    """
    计算一棵树的叶节点个数
    :param tree: 树
    :return: 个数
    """
    res = 0
    label = list(tree.keys())[0]
    next_node = tree[label]
    for feature in next_node.keys():
        if isinstance(next_node[feature], dict):
            res += get_leaf_num(next_node[feature])
        else:
            res += 1
    return res


def get_tree_depth(tree):
    """
    计算一棵树的深度
    :param tree: 树
    :return: 深度
    """
    res = 0
    label = list(tree.keys())[0]
    next_node = tree[label]
    for feature in next_node.keys():
        if isinstance(next_node[feature], dict):
            cur_depth = 1 + get_tree_depth(next_node[feature])  # 一直遍历到叶节点
        else:
            cur_depth = 1  # 递归返回1层
        if cur_depth > res:  # 记录当前节点下的最大深度
            res = cur_depth
    return res


def eval_plot(true_label: list, prob: list, eval_type='ROC', save_path=None):
    """
    画ROC图
    :param true_label: 真实的标签
    :param prob: 计算得到的概率
    :param eval_type: 绘制的图的类型
    :param save_path: 保存路径（可选）
    """
    data = zip(true_label, prob)
    sorted_data = sorted(data, key=operator.itemgetter(1), reverse=True)
    TP = FP = 0
    TPR = []
    FPR = []
    count_true = str(true_label).count("1")
    count_false = len(true_label) - count_true
    for item in sorted_data:
        if item[0] == 1:
            TP += 1
        elif item[0] == 0:
            FP += 1
        TPR.append(float(TP / count_true))
        FPR.append(float(FP / count_false))
    plt.figure()
    if eval_type == 'ROC':
        plt.plot(FPR, TPR)
        plt.plot([0, 1], [0, 1], linestyle=':')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
    elif eval_type == 'KS':
        plt.plot(FPR, label='FPR')
        plt.plot(TPR, label='TPR')
        plt.legend()
        plt.xlabel('cases num')
        plt.ylabel('rate')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
