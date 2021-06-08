from data_mining.mid_job import myDecisionTree
import numpy as np

# true_data = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1])
# prob = np.array([0.3, 0.2, 0.1, 0.4, 0.7, 0.15, 0.75, 0.65, 0.45, 0.45, 0.9, 0.8, 0.35, 0.7, 0.85, 0.9])
# labeled = np.array([0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1])

# myDecisionTree.eval_plot(true_data.tolist(), prob.tolist(), eval_type='ROC')

# tree = {'a': {0: {'b': {0: 0, 1: {'c': {0: 1, 1: 1, 2: 0}}}}, 1: {'c': {0: 1, 1: 1, 2: {'b': {0: 1, 1: 1}}}}, 2: 0}}
test_data = np.array([[1, 'dark_green', 'curl_up', 'little_heavily', 'distinct', 'sinking', 'hard_smooth'],
                      [1, 'black', 'curl_up', 'heavily', 'distinct', 'sinking', 'hard_smooth'],
                      [1, 'black', 'curl_up', 'little_heavily', 'distinct', 'sinking', 'hard_smooth'],
                      [1, 'dark_green', 'little_curl_up', 'little_heavily', 'distinct', 'little_sinking', 'soft_stick'],
                      [1, 'black', 'little_curl_up', 'little_heavily', 'little_blur', 'little_sinking', 'soft_stick'],
                      [0, 'dark_green', 'stiff', 'clear', 'distinct', 'even', 'soft_stick'],
                      [0, 'light_white', 'little_curl_up', 'heavily', 'little_blur', 'sinking', 'hard_smooth'],
                      [0, 'black', 'little_curl_up', 'little_heavily', 'distinct', 'little_sinking', 'soft_stick'],
                      [0, 'light_white', 'curl_up', 'little_heavily', 'blur', 'even', 'hard_smooth'],
                      [0, 'dark_green', 'curl_up', 'heavily', 'little_blur', 'little_sinking', 'hard_smooth'],
                      [1, 'dark_green', 'curl_up', 'heavily', 'distinct', 'sinking', 'hard_smooth'],
                      [1, 'light_white', 'curl_up', 'little_heavily', 'distinct', 'sinking', 'hard_smooth'],
                      [1, 'black', 'little_curl_up', 'little_heavily', 'distinct', 'little_sinking', 'hard_smooth'],
                      [0, 'black', 'little_curl_up', 'heavily', 'little_blur', 'little_sinking', 'hard_smooth'],
                      [0, 'light_white', 'stiff', 'clear', 'blur', 'even', 'hard_smooth'],
                      [0, 'light_white', 'curl_up', 'little_heavily', 'blur', 'even', 'soft_stick'],
                      [0, 'dark_green', 'little_curl_up', 'little_heavily', 'little_blur', 'sinking', 'hard_smooth']])
test_labels = ['label', 'color', 'root', 'knocks', 'texture', 'navel', 'touch']
# item1 = [-1, 0, 1, 0, 1]
# item2 = [-1, 1, 0, 0, 1]

# myDecisionTree.create_plot(tree)
test_tree = myDecisionTree.create_tree(test_data, test_labels, 'c45')
myDecisionTree.create_plot(test_tree)
# print(myDecisionTree.predict(test_tree, test_labels, item1))
# print(myDecisionTree.predict(test_tree, test_labels, item2))
# count = []
# myDecisionTree.count_accuracy(test_tree, test_data, test_labels, count)
# print(count)
# tn, fn, ac = myDecisionTree.accuracy(test_tree, test_data, test_labels)
# print(tn)
# print(fn)
# print(ac)
# count = []
# myDecisionTree.count_label(test_tree, test_data, test_labels, '1', count)
# print(count)
# count = []
# prob_tree = myDecisionTree.count_label(test_tree, test_data, test_labels, '1', count, True)
# myDecisionTree.create_plot(prob_tree)
pruned_tree = myDecisionTree.cut_branch_rep(test_tree, test_data, test_labels)
myDecisionTree.create_plot(pruned_tree)
