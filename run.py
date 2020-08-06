import numpy as np
import pdb
from collections import Counter
import time
import pandas as pd

class DecisionNode:
    def __init__(self, left, right, decision_function, class_label=None):
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    decision_tree_root = DecisionNode(None, None, lambda x : x[0] == 1)
    n2 = DecisionNode(None, None, lambda x: x[1] == 0)
    n3 = DecisionNode(None, None, lambda x: x[3] == 1)
    n4 = DecisionNode(None, None, lambda x: x[2] == x[3])
    one = DecisionNode(None, None, None, 1)
    zero = DecisionNode(None, None, None, 0)

    decision_tree_root.left = one
    decision_tree_root.right = n2
    n2.left = n3
    n2.right = n4
    n3.left = one
    n3.right = zero
    n4.left = one
    n4.right = zero

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    tn = np.sum((classifier_output == 0) * (true_labels == 0))
    tp = np.sum((classifier_output == 1) * (true_labels == 1))
    fn = np.sum((classifier_output == 0) * (true_labels == 1))
    fp = np.sum((classifier_output == 1) * (true_labels == 0))

    return [[tp, fn], [fp, tn]]

def precision(classifier_output, true_labels):
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    tp = np.sum((classifier_output == 1) * (true_labels == 1))
    fp = np.sum((classifier_output == 1) * (true_labels == 0))

    return tp / (tp+fp)


def recall(classifier_output, true_labels):
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    tp = np.sum((classifier_output == 1) * (true_labels == 1))
    fn = np.sum((classifier_output == 0) * (true_labels == 1))

    return tp / (tp+fn)

def accuracy(classifier_output, true_labels):
    classifier_output = np.array(classifier_output)
    true_labels = np.array(true_labels)
    return np.sum(classifier_output == true_labels) / true_labels.shape[0]

def gini_impurity(class_vector):
    class_vector = np.array(class_vector)
    p0 = np.sum(class_vector == 0) / class_vector.shape[0]
    p1 = np.sum(class_vector == 1) / class_vector.shape[0]
    return 1.0 - p0**2 - p1**2

def gini_gain(previous_classes, current_classes):
    previous_classes = np.array(previous_classes)
    p_entropy = gini_impurity(previous_classes)
    rem = 0
    for c in current_classes:
        c = np.array(c)
        if c.size == 0: continue
        rem += gini_impurity(c)*(c.shape[0]/previous_classes.shape[0])
    return p_entropy - rem

class DecisionTree:
    def __init__(self, depth_limit=float("inf")):
        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        self.root = self.__build_tree__(features, classes)

    def __build_tree__(self, features, classes, depth=0):
        def mode(classes):
            hmap = {}
            mode, max_freq = None, -1
            for c in classes:
                if c not in hmap: hmap[c] = 0
                hmap[c] += 1
            for c in hmap:
                if hmap[c] > max_freq:
                    max_freq = hmap[c]
                    mode = c
            return mode

        # Check base cases:
        if classes.size == 0:
            return None

        if np.unique(classes).size == 1 or depth == self.depth_limit:
            return DecisionNode(None, None, None, mode(classes))

        # Find best feature to split data:
        alpha_best, alpha_best_g, alpha_best_split = -1, float("-inf"), float("-inf")
        for alpha_idx, alpha in enumerate(features.T):
            alpha_min_val, alpha_max_val = np.min(alpha), np.max(alpha)
            if alpha_min_val == alpha_max_val: continue
            best_g, best_split = float("-inf"), None
            splits = np.linspace(alpha_min_val+0.001, alpha_max_val, num=100)
            for split in splits:
                n_idx, p_idx = np.where(alpha <= split), np.where(alpha > split)
                # pos_samples, neg_samples = alpha[p_idx], alpha[n_idx]
                pos_classes, neg_classes = classes[p_idx], classes[n_idx]
                g = gini_gain(classes, [pos_classes, neg_classes])
                if g > best_g:
                    best_g = g
                    best_split = split
            if best_g > alpha_best_g:
                alpha_best_g = best_g
                alpha_best = alpha_idx
                alpha_best_split = best_split
        # Split on feature alpha_best, with threshold alpha_best_split
        n_idx, p_idx = np.where(features[:, alpha_best] <= alpha_best_split), np.where(features[:, alpha_best] > alpha_best_split)
        n_features, n_classes = features[n_idx], classes[n_idx]
        p_features, p_classes = features[p_idx], classes[p_idx]
        # Build children
        n_node = self.__build_tree__(n_features, n_classes, depth+1)
        p_node = self.__build_tree__(p_features, p_classes, depth+1)
        # Return root
        return DecisionNode(n_node, p_node, lambda feature: feature[alpha_best] < alpha_best_split)

    def classify(self, features):
        class_labels = []

        for idx, feature in enumerate(features):
            class_labels.append(self.root.decide(feature))
        return class_labels


def generate_k_folds(dataset, k):
    f, c = dataset
    N, D = f.shape
    idx = np.random.permutation(N)
    f, c = f[idx], c[idx]
    folds = []
    test_size = N // k

    for fold in range(k):
        test_idx = np.arange(start=(fold + (k-1))%k * test_size, stop=(fold + (k-1))%k * test_size+test_size)
        test_feats, test_class = f[test_idx, :], c[test_idx]
        training_feats, training_class = np.delete(f, test_idx, axis=0), np.delete(c, test_idx, axis=0)
        folds.append(((training_feats, training_class),(test_feats,test_class)))

    return folds

class RandomForest:
    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.feat_map = {}

    def fit(self, features, classes):
        for tree_idx in range(self.num_trees):
            tree = DecisionTree(self.depth_limit)
            N, D = features.shape
            ex_idx = np.random.choice(N, size=int(N * self.example_subsample_rate), replace=False)
            f_idx = np.random.choice(D, size=int(D * self.attr_subsample_rate), replace=False)
            self.feat_map[tree_idx] = f_idx
            feats, labels = features[ex_idx,:][:,f_idx], classes[ex_idx]
            tree.fit(feats, labels)
            self.trees.append(tree)

    def classify(self, features):
        def mode(classes):
            hmap = {}
            mode, max_freq = None, -1
            for c in classes:
                if c not in hmap: hmap[c] = 0
                hmap[c] += 1
            for c in hmap:
                if hmap[c] > max_freq:
                    max_freq = hmap[c]
                    mode = c
            return mode

        N, D = features.shape
        classifications = np.zeros((N, self.num_trees))
        ret = np.ones((N,))
        for t_idx, tree in enumerate(self.trees):
            feat = features[:, self.feat_map[t_idx]]
            classifications[:, t_idx] = np.array(tree.classify(feat))
        for n in range(N):
            ret[n] = mode(classifications[n, :])
        return ret

class ChallengeClassifier:
    def __init__(self, num_trees=15, depth_limit=15, example_subsample_rate=0.7, attr_subsample_rate=0.7):
        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate
        self.mean_feat, self.std_feat = None, None
        self.feat_map = {}

    def fit(self, features, classes):
        self.mean_feat = np.mean(features, axis=0)
        self.std_feat = np.std(features, axis=0)
        features = (features - self.mean_feat) / self.std_feat

        for tree_idx in range(self.num_trees):
            tree = DecisionTree(self.depth_limit)
            N, D = features.shape
            ex_idx = np.random.choice(N, size=int(N * self.example_subsample_rate), replace=False)
            f_idx = np.random.choice(D, size=int(D * self.attr_subsample_rate), replace=False)
            self.feat_map[tree_idx] = f_idx
            feats, labels = features[ex_idx,:][:,f_idx], classes[ex_idx]
            tree.fit(feats, labels)
            self.trees.append(tree)

    def classify(self, features):
        def mode(classes):
            hmap = {}
            mode, max_freq = None, -1
            for c in classes:
                if c not in hmap: hmap[c] = 0
                hmap[c] += 1
            for c in hmap:
                if hmap[c] > max_freq:
                    max_freq = hmap[c]
                    mode = c
            return mode

        features = (features - self.mean_feat) / self.std_feat
        N, D = features.shape
        classifications = np.zeros((N, self.num_trees))
        ret = np.ones((N,))
        for t_idx, tree in enumerate(self.trees):
            feat = features[:, self.feat_map[t_idx]]
            classifications[:, t_idx] = np.array(tree.classify(feat))
        for n in range(N):
            ret[n] = mode(classifications[n, :])
        return list(ret)


class Vectorization:
    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        return data*data + data

    def non_vectorized_slice(self, data):
        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        row_sum = np.sum(data[0:100, :], axis=1)
        max_idx = np.argmax(row_sum)
        return (row_sum[max_idx], max_idx)

    def non_vectorized_flatten(self, data):
        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        data = data.flatten()
        data = data[data > 0]
        un, ct = np.unique(data, return_counts=True)
        return list(zip(un,ct))

def get_ord(x):
    summed = 0
    for c in x:
        summed += ord(c)

    return summed

def read_csv(filepath, class_index=-1):
    df = pd.read_csv(filepath)

    df['job'] = df['job'].apply(lambda x: get_ord(x))
    df['education'] = df['education'].apply(lambda x: get_ord(x))
    df['marital'] = df['marital'].apply(lambda x: get_ord(x))
    df['default'] = df['default'].apply(lambda x: get_ord(x))
    df['housing'] = df['housing'].apply(lambda x: get_ord(x))
    df['loan'] = df['loan'].apply(lambda x: get_ord(x))
    df['contact'] = df['contact'].apply(lambda x: get_ord(x))
    df['month'] = df['month'].apply(lambda x: get_ord(x))
    df['day_of_week'] = df['day_of_week'].apply(lambda x: get_ord(x))
    df['poutcome'] = df['poutcome'].apply(lambda x: get_ord(x))

    df = np.array(df)

    if(class_index == -1):
        classes= df[:,class_index]
        features = df[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= df[:, class_index]
        features = df[:, 1:]
        return features, classes

if __name__ == "__main__":
    cc = ChallengeClassifier(15,15,0.83,0.83)
    features, classes = read_csv('train.csv')
    cc.fit(features, classes)

    features, classes = read_csv('test.csv')
    classified = cc.classify(features)

    print(classified)
