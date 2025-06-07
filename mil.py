from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn import manifold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def draw_manifold(n_sample, t_sample):
    embed = manifold.Isomap(n_components=2)
    # embed = manifold.MDS(n_components=2, random_state=233)
    # embed = manifold.TSNE(n_components=2, random_state=233)

    trans = embed.fit_transform(np.vstack((n_sample, t_sample)))
    n_trans = trans[:n_sample.shape[0], :]
    t_trans = trans[n_sample.shape[0]:, :]
    n_num = n_trans.shape[0]
    t_num = t_trans.shape[0]
    n_trans = np.hstack((np.reshape(range(n_num), (n_num, 1)), n_trans))  # add case_cell_num
    t_trans = np.hstack((np.reshape(range(t_num), (t_num, 1)), t_trans))
    n_trans = n_trans[np.lexsort((n_trans[:, 1],))]  # sort according to x coordinate
    t_trans = t_trans[np.lexsort((t_trans[:, 1],))]
    plt.scatter(t_trans[:, 1], t_trans[:, 2], c='red')
    plt.scatter(n_trans[:, 1], n_trans[:, 2], c='blue')
    # plt.xlim([-20, 35])
    # plt.ylim([-25, 20])
    plt.show()


def draw_kde(n_sample, t_sample, n_all, t_all):
    embed = manifold.Isomap(n_components=2).fit(np.vstack((n_sample, t_sample)))
    n_all_trans = embed.transform(n_all)
    t_all_trans = embed.transform(t_all)
    for all_trans in (n_all_trans, t_all_trans, np.vstack((n_all_trans, t_all_trans))):
        sns.kdeplot(x=all_trans[:, 0], y=all_trans[:, 1], fill=True, cmap='Spectral', cbar=True)
        plt.xlim([-20, 35])
        plt.ylim([-25, 20])
        plt.show()


def draw_scatter_with_label(x, y, case_nu=None, case_lab=None, emphasis=None):
    """
    :param x: feature values
    :param y: labels of each instance
    :param case_nu: case to which each instance belongs (None: not drawn boundary)
    :param case_lab: labels of each case (None: not draw boundary)
    :param emphasis: idx of case in case_lab to draw for emphasis (case and case_lab can't be None)
    """
    colors = ['blue', 'red']

    if case_nu is None or case_lab is None:
        plt.scatter(x[y == 1, 0], x[y == 1, 1], c='red', alpha=0.2)
        plt.scatter(x[y == 0, 0], x[y == 0, 1], c='blue', alpha=0.2)
    else:
        case_lab = np.int32(case_lab)
        case_nu = np.int32(case_nu)
        case_unique = pd.unique(case_nu)
        _case_count = case_unique.size
        for i in range(case_lab.size):
            case_idx = case_unique[i]
            if emphasis is None or i == emphasis:
                alpha = 1.0
            else:
                alpha = 0.05
            case_x = x[case_nu == case_idx, :]
            case_y = y[case_nu == case_idx]
            plt.scatter(case_x[case_y == 1, 0], case_x[case_y == 1, 1], c='red', alpha=alpha)
            plt.scatter(case_x[case_y == 0, 0], case_x[case_y == 0, 1], c='blue', alpha=alpha)
            rect = plt.Rectangle((np.min(case_x[:, 0]), np.min(case_x[:, 1])),
                                 np.max(case_x[:, 0]) - np.min(case_x[:, 0]),
                                 np.max(case_x[:, 1]) - np.min(case_x[:, 1]),
                                 edgecolor=colors[case_lab[i]],
                                 facecolor='none', alpha=alpha)
            plt.gca().add_patch(rect)
    plt.xlim([-75, 125])
    plt.ylim([-45, 40])
    plt.show()


def mil_train(train_X, train_idx, train_case_label, train_inst_label):  # as mi-SVM
    # the first SVC to initiate each instance's label
    clf = NuSVC(nu=0.5, kernel='linear')
    X_MI = np.zeros((train_case_label.size, train_X.shape[1]))
    Y_MI = np.zeros(train_case_label.size)
    for i in range(train_case_label.size):
        case_X_train = train_X[train_idx[i]:train_idx[i + 1], :]
        if train_case_label[i] == 0:
            case_X_MI = np.mean(case_X_train, axis=0)  # negative bag: label its centroid as negative
        else:
            signi_idx = np.argmax(np.linalg.norm(case_X_train, axis=1))
            case_X_MI = case_X_train[signi_idx, :]  # positive bag: label the most positive instance as positive
        X_MI[i, :] = case_X_MI
        Y_MI[i] = train_case_label[i]
    clf.fit(X_MI, Y_MI)
    train_Y = clf.predict(train_X)

    pred_inst_old = np.zeros(train_Y.size)
    iteration = 1
    # train classifier
    while True:
        # instance prediction -> case prediction
        case_pred_train = np.array([bool(np.sum(train_Y[train_idx[i]:train_idx[i + 1]]))
                                    for i in range(train_case_label.size)])
        case_pred_train = np.double(case_pred_train)

        train_Y[train_inst_label == 0] = 0  # labels of instance in negative cases should be negative

        if len(np.where((1 - case_pred_train) * train_case_label == 1)[
                   0]) > 0:  # a positive case is classified as negative
            prob = clf.decision_function(train_X)
            locs = np.where((1 - case_pred_train) * train_case_label == 1)[0]
            for loc in locs:  # point with the largest discrimination is assigned as positive
                prob_case = prob[train_idx[loc]:train_idx[loc + 1]]
                prob_case_argmax = np.argmax(prob_case)
                train_Y[train_idx[loc] + prob_case_argmax] = 1

        if iteration == 1:
            # clf = SVC(C=0.1, kernel='poly')
            clf = LogisticRegression(max_iter=1000, C=0.01, solver='lbfgs')
        clf.fit(train_X, train_Y)  # fit after modifying prediction
        pred_inst_new = clf.predict(train_X)

        if (pred_inst_new == pred_inst_old).all():
            # update prediction and case, stop iteration
            break
        else:
            pred_inst_old = pred_inst_new

        train_Y = pred_inst_new.copy()  # prediction as well as new label
        iteration += 1

    case_pred_train = np.array(
        [bool(np.sum(train_Y[train_idx[i]:train_idx[i + 1]])) for i in range(train_case_label.size)])
    case_pred_train = np.double(case_pred_train)
    print('Training Accuracy:' + str(np.sum(case_pred_train == train_case_label) / train_case_label.size))

    # train_Y_prob = clf.predict_proba(train_X)[:, 1]
    # train_case_prob = np.array(
    #     [np.max(train_Y_prob[train_idx[i]:train_idx[i + 1]]) for i in range(train_case_label.size)])

    # return train_Y, clf, train_case_prob
    return clf


def mil_v2_train(clf, train_X, train_idx, train_case_label, train_inst_label, k=1, max_iters=100):  # as MI-SVM
    train_Y = initiate_labels(train_X, train_inst_label)
    # 2.4: C=0.1,solver='liblinear'
    clf.fit(train_X, train_Y)

    train_mask_old = np.zeros(train_X.shape[0], dtype='bool')
    iteration = 1
    # train classifier
    while True:
        # update instances and labels for training
        train_mask_new = np.zeros(train_X.shape[0], dtype='bool')  # select partial instances (bool=True) for training
        for i in range(train_idx.size - 1):  # the ith case
            if train_case_label[i] == 0:
                train_Y[train_idx[i]:train_idx[i + 1]] = 0  # rectify labels
                train_mask_new[train_idx[i]:train_idx[i + 1]] = True  # include all instances in each negative bag
            else:
                prob_Y = clf.decision_function(train_X)
                prob_Y_case = prob_Y[train_idx[i]:train_idx[i + 1]]
                prob_Y_case_argmax_k = np.argsort(prob_Y_case)[-k:]  # most significant instance in each positive bag
                for prob_Y_case_argmax in prob_Y_case_argmax_k:
                    train_Y[train_idx[i] + prob_Y_case_argmax] = 1  # the same with negative bags
                    train_mask_new[train_idx[i] + prob_Y_case_argmax] = True

        # if iteration == 1:
        #     # clf = SVC(C=1.0, kernel='rbf')
        #     clf = LogisticRegression(max_iter=1000, C=0.01, solver='liblinear')

        if (train_mask_new == train_mask_old).all():  # check whether to end iteration
            break
        else:
            clf.fit(train_X[train_mask_new, :], train_Y[train_mask_new])
            train_mask_old = train_mask_new
        iteration += 1
        if iteration > max_iters:  # converge failed, stop iteration
            break

    train_Y = clf.predict(train_X)  # update all labels
    train_case_pred = np.array(
        [bool(np.sum(train_Y[train_idx[i]:train_idx[i + 1]])) for i in range(train_case_label.size)])
    train_case_pred = np.double(train_case_pred)
    print('Training Accuracy:' + str(np.sum(train_case_pred == train_case_label) / train_case_label.size))

    # train_Y_prob = clf.predict_proba(train_X)[:, 1]
    # train_case_prob = np.array(
    #     [np.max(train_Y_prob[train_idx[i]:train_idx[i + 1]]) for i in range(train_case_label.size)])

    return clf


def mi_train(clf, train_X, train_idx, train_case_label, train_inst_label, max_iters=100):  # as mi-SVM with new configs
    train_Y = initiate_labels(train_X, train_inst_label)
    clf.fit(train_X, train_Y)

    pred_inst_old = np.zeros(train_Y.size)
    iteration = 1
    # train classifier
    while True:
        # instance prediction -> case prediction
        case_pred_train = np.array([bool(np.sum(train_Y[train_idx[i]:train_idx[i + 1]]))
                                    for i in range(train_case_label.size)]).astype(np.double)
        train_Y[train_inst_label == 0] = 0  # labels of instance in negative cases should be negative
        # if there is a positive case classified as negative
        if len(np.where((1 - case_pred_train) * train_case_label == 1)[0]) > 0:
            prob_Y = clf.decision_function(train_X)
            locs = np.where((1 - case_pred_train) * train_case_label == 1)[0]  # which cases are misclassified
            for loc in locs:  # point with the largest probability is assigned as positive
                prob_case = prob_Y[train_idx[loc]:train_idx[loc + 1]]
                prob_case_argmax = np.argmax(prob_case)
                train_Y[train_idx[loc] + prob_case_argmax] = 1

        clf.fit(train_X, train_Y)  # fit after modifying prediction
        pred_inst_new = clf.predict(train_X)

        if (pred_inst_new == pred_inst_old).all():  # 2 pred are the same
            break
        else:
            pred_inst_old = np.copy(pred_inst_new)
            train_Y = np.copy(pred_inst_new)  # prediction this time -> rectify -> training next time

        iteration += 1
        if iteration > max_iters:  # converge failed, stop iteration
            break

    train_Y = clf.predict(train_X)  # update all labels
    train_case_pred = np.array(
        [bool(np.sum(train_Y[train_idx[i]:train_idx[i + 1]])) for i in range(train_case_label.size)])
    train_case_pred = np.double(train_case_pred)
    print('Training Accuracy:' + str(np.sum(train_case_pred == train_case_label) / train_case_label.size))

    return clf


def initiate_labels(train_X, train_inst_label):
    # split instances in positive bags into positive and negative (pseudo-label)
    train_Y = train_inst_label.copy()
    positive_bags_inst = train_X[train_inst_label == 1, :]
    neigh = NearestNeighbors(n_neighbors=10, radius=0.4).fit(train_X)
    _, indices = neigh.kneighbors(positive_bags_inst)
    pos_locs = np.where(train_inst_label == 1)[0]
    for i in range(pos_locs.size):
        indices_i = indices[i, :]
        pos_neigh_num = np.sum(train_inst_label[indices_i])
        if pos_neigh_num > 8:
            train_Y[pos_locs[i]] = 1.0
        else:
            train_Y[pos_locs[i]] = 0.0
    return train_Y


def draw_prob_distri(prob, case, label, test, margin):
    """
    :param prob: probabilities of each point
    :param label: label of each case
    :param case: which case each point belongs to
    :param test: which case is being tested
    :param margin: indices of margin cases
    """
    label_ = label.copy().astype(np.float32)
    for m_idx in margin:
        locs = np.where(case == m_idx)[0].astype(int)
        label_[locs] = 0.5 * label_[locs] + 0.25
    plt.figure(figsize=(12, 5))
    plt.scatter(case, prob, c=label_)
    for c in np.unique(case):
        c_max = np.max(prob[case == c])
        plt.scatter(c, c_max, c='red')
    plt.plot((test, test), (0, 1), c='orange', linestyle='--')
    plt.plot((0, np.max(case)), (0.5, 0.5), c='blue', linestyle='--')
    plt.show()


"""ADJUST"""
csv_name = "./feature/featureData_2.15_clean_twoFalseNegatives.csv"
# csv_name = "./feature/featureData_2.15_ExplicitFeatures.csv"
featureData = pd.read_csv(csv_name)

case_num = featureData.iloc[:, 4].values
Y_bag = featureData.iloc[:, 5].values  # label of bag to which each instance belongs
X = featureData.iloc[:, 6:518].values  # case nums, case labels and feature vectors
# X = featureData.iloc[:, 6:18].values  # all explicit features
# X = featureData.iloc[:, 6:11].values  # explicit morphological features
# X = featureData.iloc[:, 11:18].values  # explicit componential features

scaler = StandardScaler()
# X = scaler.fit_transform(X)

case_count, idx = np.unique(case_num, return_index=True)  # idx : index of 1st instance of each case
case_count = int(np.max(case_count) + 1)  # case_count : total number of cases
case_label = Y_bag[idx]  # case_label : label of each case
idx = np.append(idx, Y_bag.size - 1)  # append idx with total number of instances to facilitate loading the last case

"""ADJUST"""
N_sample = X[idx[0]:idx[1], :]  # 1833417
T_sample = X[idx[46]:idx[47], :]  # 1955514
M_case_idx = list(range(7, 24)) + list(range(52, 62))  # (27) neck margin cases
NT_case_idx = list(range(0, 7)) + list(range(24, 52))  # (35) normal and tumor cases
# S_case_idx = list(range(24, 30)) + list(range(68, 76))  # (14) SMA margin cases

# to_remove = [28, 37, 39, 40, 41, 42, 43]
# for val in to_remove:
#     NT_case_idx.remove(val)

M_case_label = case_label[M_case_idx]
NT_case_label = case_label[NT_case_idx]
# S_case_label = case_label[S_case_idx]

# embedding = manifold.Isomap(n_components=2).fit(np.vstack((N_sample, T_sample)))
# X_trans = embedding.transform(X)  # Isomap embedding of all instances, modeled by two samples
X_trans = manifold.Isomap(n_components=2).fit_transform(X)
# draw_scatter_with_label(X_trans, Y_bag)

# # leave one out validation for neck/SMA margin
test_case_idx = M_case_idx
test_case_label = M_case_label

case_prob_test = np.zeros(len(test_case_idx))
case_pred_test = np.zeros(len(test_case_idx))
for fold in range(len(test_case_idx)):
    print('Fold:' + str(fold))
    case_idx_test = np.array([test_case_idx[fold]])
    case_label_test = np.array([test_case_label[fold]])

    case_idx_train = np.array([test_case_idx[j] for j in range(len(test_case_idx)) if j != fold])
    case_idx_train = np.append(case_idx_train, NT_case_idx)  # add all NT indices to training indices when test M

    # case_idx_train = np.array(NT_case_idx)
    # """replace with above two rows"""

    # if test_type == 'S':
    #     case_idx_train = np.append(case_idx_train, M_case_idx)  # add all M indices to training indices when test S
    case_label_train = case_label[case_idx_train]

    X_train = np.vstack(list((X[idx[i]:idx[i + 1]] for i in case_idx_train)))  # extract training/test set by indices
    X_test = np.vstack(list((X[idx[i]:idx[i + 1]] for i in case_idx_test)))

    idx_train = np.zeros(case_idx_train.size + 1,
                         dtype='int')  # indices of 1st instance of each case in training/test set
    idx_test = np.zeros(case_idx_test.size + 1, dtype='int')
    inst_label_train = np.zeros(X_train.shape[0])  # instance labels in training/test set, equal to belonging case
    inst_label_test = np.zeros(X_test.shape[0])
    for i in range(1, case_idx_train.size + 1):
        idx_train[i] = idx_train[i - 1] + idx[case_idx_train[i - 1] + 1] - idx[case_idx_train[i - 1]]
        inst_label_train[idx_train[i - 1]:idx_train[i]] = case_label_train[i - 1]
    for i in range(1, case_idx_test.size + 1):
        idx_test[i] = idx_test[i - 1] + idx[case_idx_test[i - 1] + 1] - idx[case_idx_test[i - 1]]
        inst_label_test[idx_test[i - 1]:idx_test[i]] = case_label_test[i - 1]

    case_num_train = np.zeros(X_train.shape[0])  # case to which each instance belongs in training/test set
    case_num_test = np.zeros(X_test.shape[0])
    for i in range(idx_train.size - 1):
        case_num_train[idx_train[i]:idx_train[i + 1]] = case_idx_train[i]
    for i in range(idx_test.size - 1):
        case_num_test[idx_test[i]:idx_test[i + 1]] = case_idx_test[i]

    classifier = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    # classifier = SVC(C=1.0, kernel='linear', probability=True, random_state=233)
    # StandardScaler: 'liblinear'
    # classifier = mi_train(classifier, X_train, idx_train,
    #                       case_label_train, inst_label_train)
    classifier = mil_v2_train(classifier, X_train, idx_train,
                              case_label_train, inst_label_train, k=1)

    # coef = classifier.coef_
    # coef_norm = coef/np.mean(abs(coef))
    # if fold == 0:
    #     sort_idx = np.argsort(coef_norm[0, :])
    #     sorted_coef_norm = np.sort(coef_norm)
    # else:
    #     sorted_coef_norm = np.expand_dims(coef_norm[0, sort_idx], 0)  # sorted according to the first fold
    # sorted_coef_norm = sorted_coef_norm.T

    prob_test = classifier.predict_proba(X_test)[:, 1]
    Y_test = classifier.predict(X_test)
    case_pred_test[fold] = np.double(bool(np.sum(Y_test)))
    case_prob_test[fold] = np.max(prob_test)

    # draw_prob_distri(classifier.predict_proba(X)[:, 1], case_num, Y_bag, test_case_idx[fold], test_case_idx)
    print('Test probability:' + str(case_prob_test[fold]))
    print('Test prediction:' + str(case_pred_test[fold]))
    print('Test label:' + str(case_label_test[0]))

acc = np.sum(case_pred_test == test_case_label) / test_case_label.size
sens = np.sum((case_pred_test == test_case_label) * (test_case_label == 1)) / np.sum(test_case_label == 1)
spec = np.sum((case_pred_test == test_case_label) * (test_case_label == 0)) / np.sum(test_case_label == 0)
print('Test accuracy:' + str(acc))
print('Test sensitivity' + str(sens))
print('Test specificity' + str(spec))
auc = roc_auc_score(test_case_label, case_prob_test)
print('Test AUC:' + str(auc))
# fpr, tpr, thresholds = roc_curve(M_case_label, case_prob_test, pos_label=1)
# print('Optimal threshold:' + str(thresholds[np.argmax(tpr-fpr)]))

# featureData['trained_label'] = classifier.predict(X)
# featureData.to_csv("./feature/featureData_2.15_1.csv")
pass
