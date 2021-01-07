# -*- coding: utf-8 -*-

import numpy as np
from gcforest.gcforest import GCForest
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_curve, f1_score, recall_score, precision_score
import xlrd

def load_csv_data(filename):
    data = []
    datafile = open(filename)
    for line in datafile:
        fields = line.strip().split('\t')
        data.append([float(field) for field in fields])
    data = np.array(data)
    return data

def get_toy_config():
    config = {}
    ca_config = {}
    ca_config["random_state"] = 0
    ca_config["max_layers"] = 100
    ca_config["early_stopping_rounds"] = 3
    ca_config["n_classes"] = 2
    ca_config["estimators"] = []
    ca_config["estimators"].append(
            {"n_folds": 5, "type": "LogisticRegression"} )
    ca_config["estimators"].append(
        {"n_folds": 5, "type": "XGBClassifier", "n_estimators": 300, "max_depth": 5,
         "objective": "multi:softprob", "silent": True, "nthread": -1, "learning_rate": 0.1,"num_class":2})
    ca_config["estimators"].append({"n_folds": 5, "type": "RandomForestClassifier", "n_estimators": 300, "max_depth": None, "n_jobs": -1})
    ca_config["estimators"].append({"n_folds": 5, "type": "ExtraTreesClassifier", "n_estimators": 300, "max_depth": None, "n_jobs": -1})
    config["cascade"] = ca_config
    return config

def Sample_generation(lncRNAFeature,proteinFeature,interaction):
    positive_feature = []
    negative_feature = []
    for i in range(np.shape(interaction)[0]):
        for j in range(np.shape(interaction)[1]):
            temp = np.append(lncRNAFeature[i], proteinFeature[j])
            if int(interaction[i][j]) == 1:
                positive_feature.append(temp)
            elif int(interaction[i][j]) == 0:
                negative_feature.append(temp)
    negative_sample_index = np.random.choice(np.arange(len(negative_feature)), size = len(positive_feature),
                                             replace = False)
    negative_sample_feature = []
    for i in negative_sample_index:
        negative_sample_feature.append(negative_feature[i])
    feature = np.vstack((positive_feature, negative_sample_feature))
    label1 = np.ones((len(positive_feature), 1))
    label0 = np.zeros((len(negative_sample_feature), 1))
    label = np.vstack((label1, label0))
    return feature,label

def excle2matrix(path):
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows
    ncols = table.ncols
    datamatrix = np.zeros((nrows, ncols))

    for x in range(ncols):
        cols = table.col_values(x)
        cols1 = np.matrix(cols)
        datamatrix[:, x] = cols1
    return datamatrix


if __name__ == '__main__':
    # Feature extracted from seq.
    lncRNAFeature = excle2matrix('./datasets/3RNAseq.xlsx')
    proteinFeature = excle2matrix('./datasets/3Proteinseq.xlsx')
    interaction = excle2matrix('./datasets/3LPI.xlsx')
    feature, label = Sample_generation(lncRNAFeature,proteinFeature,interaction)

    rs = np.random.randint(0, 1000, 1)[0]
    kf = KFold(n_splits = 5, shuffle = True)

    test_auc_fold = []
    test_aupr_fold = []
    test_acc_fold = []
    test_f1_fold = []
    test_mcc_fold = []
    test_recall_fold = []
    test_precision_fold = []
    for train_index, test_index in kf.split(label[:, 0]):
        Xtrain, Xtest = feature[train_index], feature[test_index]
        Ytrain, Ytest = label[train_index], label[test_index]

        config = get_toy_config()
        rf = GCForest(config)
        Ytrain = Ytrain.flatten()
        rf.fit_transform(Xtrain, Ytrain)

        # deep forest
        predict_y = rf.predict(Xtest)
        acc = accuracy_score(Ytest, predict_y)
        print("Test Accuracy of GcForest = {:.4f} %".format(acc * 100))
        prob_predict_y = rf.predict_proba(Xtest)  # Give a result with probability values，the probability sum is 1
        predictions_validation = prob_predict_y[:, 1]
        fpr, tpr, auc_thresholds = roc_curve(Ytest, predictions_validation)
        roc_auc = auc(fpr, tpr)
        prec, rec, pr_threshods = precision_recall_curve(Ytest, predictions_validation)
        aupr = auc(rec, prec)
        f1 = f1_score(Ytest, predict_y)
        precision = precision_score(Ytest, predict_y)
        recall = recall_score(Ytest, predict_y)

        print("Test roc_auc of GcForest = {:.4f} %".format(roc_auc * 100))
        print("Test aupr of GcForest = {:.4f} %".format(aupr * 100))

        test_auc_fold.append(roc_auc)
        test_aupr_fold.append(aupr)
        test_acc_fold.append(acc)
        test_f1_fold.append(f1)
        test_recall_fold.append(recall)
        test_precision_fold.append(precision)

    mean_auc = np.mean(test_auc_fold)
    mean_pr = np.mean(test_aupr_fold)
    mean_f1 = np.mean(test_f1_fold)
    mean_recall = np.mean(test_recall_fold)
    mean_acc = np.mean(test_acc_fold)
    mean_precision = np.mean(test_precision_fold)

    std_auc = np.std(test_auc_fold)
    std_aupr = np.std(test_aupr_fold)
    std_precision = np.std(test_precision_fold)
    std_acc = np.std(test_acc_fold)
    std_recall = np.std(test_recall_fold)
    std_f1 = np.std(test_f1_fold)
    std_MCC = np.std(test_mcc_fold)

    print('mean auc aupr', mean_auc, mean_pr)