-
#数据标准化处理
from sklearn.preprocessing import StandardScaler

#使用SMOTE包解决样本不平衡的问题
from imblearn.over_sampling import SMOTE                              

#导入训练测试数据拆分
from sklearn.model_selection import train_test_split
from collections import Counter

#引入指标检验计算结果
from sklearn.metrics import classification_report                    
from sklearn.metrics import accuracy_score,precision_score,recall_score,fbeta_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_auc_score

#导入算法模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#网格搜索法调参
from sklearn.model_selection import KFold,cross_val_score
from sklearn.model_selection import GridSearchCV

#训练集与测试集数据集划分
x_train,x_test,y_train,y_test = train_test_split(x_oversample,y_oversample,random_state=4,test_size=0.2)

作者：秋旻
链接：https://zhuanlan.zhihu.com/p/141647157
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#实例化模型
logreg = LogisticRegression(random_state=7)

#需要优化的参数取值 C为正则化力度  penalty
param_test = {'C':[0.01,0.1,1,10],'penalty':['l1','l2']}

#利用GridSearchCV自动调参
grid_search_lr = GridSearchCV(logreg,param_grid=param_test,scoring='recall',cv=5)
GridSearchCV(cv=5, error_score='raise',
       estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=7, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring='recall', verbose=0)

#找到最佳分数下的最佳模型参数
print(grid_search_lr.best_score_)
print(grid_search_lr.best_params_)

#得到最优参数的模型
lrmodel = LogisticRegression(C=10,penalty='l2',random_state=7)
#训练模型
lrmodel.fit(x_train,y_train)

#得到模型预测值
y_pred = lrmodel.predict(x_test)

print('Logreg classification\n')
#查看混淆矩阵
print('confusion matrix\n',confusion_matrix(y_test,y_pred))

#查看分类报告
print('classification report\n',classification_report(y_test,y_pred))
#查看预测精度与ROC_AUC曲线
print('Accuracy:',accuracy_score(y_test,y_pred))
print('Area under the curve:',roc_auc_score(y_test,y_pred))

#交叉验证分数
print('Cross Validation of x and y Train:')
print(cross_val_score(lrmodel,x_train,y_train,cv=5,scoring='recall'))


作者：秋旻
链接：https://zhuanlan.zhihu.com/p/141647157
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

#建立随机森林分类器
RFclassifier = RandomForestClassifier(random_state=7)

#模型的调参和验证
param_test2 = {'max_depth':[3,5,None],'n_estimators':[5,8,10],'max_features':[5,6,7,8]}

#利用GridSearchCV自动调参
grid_search_RFC = GridSearchCV(RFclassifier,param_grid=param_test2,cv=5,scoring='recall')
GridSearchCV(cv=5, error_score='raise',
       estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            oob_score=False, random_state=7, verbose=0, warm_start=False),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'max_depth': [3, 5, None], 'n_estimators': [5, 8, 10], 'max_features': [5, 6, 7, 8]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring='recall', verbose=0)
#找到最佳分数下的最佳模型参数
print(grid_search_RFC.best_score_)
print(grid_search_RFC.best_params_)

#得到最优参数的模型
RFCmodel = RandomForestClassifier(max_depth=None,max_features=6,n_estimators=10,random_state=7)
#训练模型
RFCmodel.fit(x_train,y_train)

#得到模型预测值
y_pred = RFCmodel.predict(x_test)


y_pred_prob = RFCmodel.predict_proba(x_test)[:,1]

precision,recall,thresholds = precision_recall_curve(y_test,y_pred_prob)

#绘制Precision Recall Curve
plt.plot(precision,recall)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()

#根据特征重要性按权重绘制图片
features = list(df_credit)[0:-1]
plt.figure(figsize=(12,6))

feat_import = pd.DataFrame({'Feature':features,'Feature importance':RFCmodel.feature_importances_})
feat_import = feat_import.sort_values(by='Feature importance',ascending=False)

g = sns.barplot(x='Feature',y='Feature importance',data=feat_import)
g.set_xticklabels(g.get_xticklabels(),rotation = 90)
g.set_title('Feature importance-RandomForest',fontsize = 20)
plt.show()
