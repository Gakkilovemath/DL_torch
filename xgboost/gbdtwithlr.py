from scipy.sparse.construct import hstack
from sklearn.model_selection import train_test_split
from sklearn.datasets.svmlight_format import load_svmlight_file
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics.ranking import roc_auc_score
from sklearn.preprocessing.data import OneHotEncoder
import numpy as np

def gbdt_lr_train(libsvmFileName):

    # load样本数据
    X_all, y_all = load_svmlight_file(libsvmFileName)

    # 训练/测试数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.3, random_state = 42)

    # 定义GBDT模型
    gbdt = GradientBoostingClassifier(n_estimators=40, max_depth=3, verbose=0,max_features=0.5)

    # 训练学习
    gbdt.fit(X_train, y_train)

    # 预测及AUC评测
    y_pred_gbdt = gbdt.predict_proba(X_test.toarray())[:, 1]
    gbdt_auc = roc_auc_score(y_test, y_pred_gbdt)
    print('gbdt auc: %.5f' % gbdt_auc)

    # lr对原始特征样本模型训练
    lr = LogisticRegression()
    lr.fit(X_train, y_train)    # 预测及AUC评测
    y_pred_test = lr.predict_proba(X_test)[:, 1]
    lr_test_auc = roc_auc_score(y_test, y_pred_test)
    print('基于原有特征的LR AUC: %.5f' % lr_test_auc)

    # GBDT编码原有特征
    X_train_leaves = gbdt.apply(X_train)[:,:,0]
    X_test_leaves = gbdt.apply(X_test)[:,:,0]

    # 对所有特征进行ont-hot编码
    (train_rows, cols) = X_train_leaves.shape

    gbdtenc = OneHotEncoder()
    X_trans = gbdtenc.fit_transform(np.concatenate((X_train_leaves, X_test_leaves), axis=0))

    # 定义LR模型
    lr = LogisticRegression()
    # lr对gbdt特征编码后的样本模型训练
    lr.fit(X_trans[:train_rows, :], y_train)
    # 预测及AUC评测
    y_pred_gbdtlr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    gbdt_lr_auc1 = roc_auc_score(y_test, y_pred_gbdtlr1)
    print('基于GBDT特征编码后的LR AUC: %.5f' % gbdt_lr_auc1)

    # 定义LR模型
    lr = LogisticRegression(n_jobs=-1)
    # 组合特征
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])

    print(X_train_ext.shape)
    # lr对组合特征的样本模型训练
    lr.fit(X_train_ext, y_train)

    # 预测及AUC评测
    y_pred_gbdtlr2 = lr.predict_proba(X_test_ext)[:, 1]
    gbdt_lr_auc2 = roc_auc_score(y_test, y_pred_gbdtlr2)
    print('基于组合特征的LR AUC: %.5f' % gbdt_lr_auc2)


if __name__ == '__main__':
    gbdt_lr_train('data/sample_libsvm_data.txt')

# 2. XGboost + LR
# python3.5.3 + scikit-learn0.18.1 + xgboost0.6

import xgboost as xgb
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.externals import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.preprocessing.data import OneHotEncoder


def xgboost_lr_train(libsvmFileNameInitial):

    # load样本数据
    X_all, y_all = load_svmlight_file(libsvmFileNameInitial)

    # 训练/测试数据分割
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size = 0.3, random_state = 42)

    # 定义xgb模型
    xgboost = xgb.XGBClassifier(nthread=4, learning_rate=0.08,
                            n_estimators=50, max_depth=5, gamma=0, subsample=0.9, colsample_bytree=0.5)
    # 训练xgb学习
    xgboost.fit(X_train, y_train)

    # 预测xgb及AUC评测
    y_pred_test = xgboost.predict_proba(X_test)[:, 1]
    xgb_test_auc = roc_auc_score(y_test, y_pred_test)
    print('xgboost test auc: %.5f' % xgb_test_auc)

    # xgboost编码原有特征
    X_train_leaves = xgboost.apply(X_train)
    X_test_leaves = xgboost.apply(X_test)


    # 合并编码后的训练数据和测试数据
    All_leaves = np.concatenate((X_train_leaves, X_test_leaves), axis=0)
    All_leaves = All_leaves.astype(np.int32)

    # 对所有特征进行ont-hot编码
    xgbenc = OneHotEncoder()
    X_trans = xgbenc.fit_transform(All_leaves)

    (train_rows, cols) = X_train_leaves.shape

    # 定义LR模型
    lr = LogisticRegression()
    # lr对xgboost特征编码后的样本模型训练
    lr.fit(X_trans[:train_rows, :], y_train)
    # 预测及AUC评测
    y_pred_xgblr1 = lr.predict_proba(X_trans[train_rows:, :])[:, 1]
    xgb_lr_auc1 = roc_auc_score(y_test, y_pred_xgblr1)
    print('基于Xgb特征编码后的LR AUC: %.5f' % xgb_lr_auc1)

    # 定义LR模型
    lr = LogisticRegression(n_jobs=-1)
    # 组合特征
    X_train_ext = hstack([X_trans[:train_rows, :], X_train])
    X_test_ext = hstack([X_trans[train_rows:, :], X_test])

    # lr对组合特征的样本模型训练
    lr.fit(X_train_ext, y_train)

    # 预测及AUC评测
    y_pred_xgblr2 = lr.predict_proba(X_test_ext)[:, 1]
    xgb_lr_auc2 = roc_auc_score(y_test, y_pred_xgblr2)
    print('基于组合特征的LR AUC: %.5f' % xgb_lr_auc2)

if __name__ == '__main__':
    xgboost_lr_train("data/sample_libsvm_data.txt")
