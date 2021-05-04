import myDecisionTree
import numpy as np
import math

# true_data = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1])
# prob = np.array([0.3, 0.2, 0.1, 0.4, 0.7, 0.15, 0.75, 0.65, 0.45, 0.45, 0.9, 0.8, 0.35, 0.7, 0.85, 0.9])
# labeled = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1])

# myDecisionTree.eval_plot(true_data.tolist(), prob.tolist(), eval_type='ROC')

# tree = {'a': {0: {'b': {0: 0, 1: {'c': {0: 1, 1: 1, 2: 0}}}}, 1: {'c': {0: 1, 1: 1, 2: {'b': {0: 1, 1: 1}}}}, 2: 0}}

test_data = np.array([[1, 0, 0, 1, 0], [0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [1, 1, 0, 1, 0],
                      [1, 0, 1, 1, 1], [1, 0, 1, 0, 1], [1, 0, 0, 1, 1], [1, 0, 0, 0, 1]])
test_labels = ['label', 'a', 'b', 'c', 'd']
# item1 = [-1, 0, 1, 0, 1]
# item2 = [-1, 1, 0, 0, 1]

# myDecisionTree.create_plot(tree)
test_tree = myDecisionTree.create_tree(test_data, test_labels, 'id3')
myDecisionTree.create_plot(test_tree)
# print(myDecisionTree.predict(test_tree, test_labels, item1))
# print(myDecisionTree.predict(test_tree, test_labels, item2))
count = []
# myDecisionTree.count_accuracy(test_tree, test_data, test_labels, count)
# print(count)
# tn, fn, ac = myDecisionTree.accuracy(test_tree, test_data, test_labels)
# print(tn)
# print(fn)
# print(ac)
# myDecisionTree.count_label(test_tree, test_data, test_labels, '1', count)
# print(count)
prob_tree = myDecisionTree.count_label(test_tree, test_data, test_labels, '1', count, True)
myDecisionTree.create_plot(prob_tree)
