# ================基于Scikit-learn接口的回归================
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import numpy as np
import seaborn as sns
import shap

X = pd.DataFrame(pd.read_csv("111/AE-train_x.csv")).values
y = pd.DataFrame(pd.read_csv("111/AE-train_y.csv")).values

#X = pd.DataFrame(pd.read_csv("train_x.csv")).values
#y = pd.DataFrame(pd.read_csv("train_y.csv")).values

X_test = pd.DataFrame(pd.read_csv("test_x.csv")).values
y_test = pd.DataFrame(pd.read_csv("test_y.csv")).values

# # 参数调优
# cv_params = {'max_depth': [5, 6, 7, 9], 'gamma':[0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1],\
#  'learning_rate': [0.01, 0.015, 0.025, 0.05], 'subsample':[0.6, 0.7, 0.8, 0.9, 1],\
#              'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1]}
# other_params = {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 400, 'gamma': 0,'min_child_weight': 1,
#                 'subsample': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.05, 'reg_lambda': 0.1, 'seed': 0}
# model = xgb.XGBRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
# optimized_GBM.fit(X, y)
# evalute_result = optimized_GBM.cv_results_
# print('每轮迭代运行结果：{0}'.format(evalute_result))
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

model = xgb.XGBRegressor(max_depth=6,          # 每一棵树最大深度，默认6；
                        learning_rate=0.05,      # 学习率，每棵树的预测结果都要乘以这个学习率，默认0.3；
                        n_estimators=400,        # 使用多少棵树来拟合，也可以理解为多少次迭代。默认100；
                        objective='reg:squarederror',   # 此默认参数与 XGBClassifier 不同
                        booster='gbtree',         # 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。默认为gbtree
                        gamma=0,                 # 叶节点上进行进一步分裂所需的最小"损失减少"。默认0；
                        min_child_weight=1,      # 可以理解为叶子节点最小样本数，默认1；
                        subsample=1,              # 训练集抽样比例，每次拟合一棵树之前，都会进行该抽样步骤。默认1，取值范围(0, 1]
                        colsample_bytree=0.8,       # 每次拟合一棵树之前，决定使用多少个特征，参数默认1，取值范围(0, 1]。
                        reg_alpha=0.05,             # 默认为0，控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
                        reg_lambda=0.1,            # 默认为1，控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                        random_state=0)
model.fit(X,y)

# 对测试集进行预测
ans = model.predict(X_test)
y_test = np.array(y_test)
ans = np.array(ans)
ans = np.around(ans, decimals=2)
y_test = np.squeeze(y_test)
absolute_errors = np.abs(ans-y_test)
absolute_errors = pd.DataFrame(absolute_errors)
ans = pd.DataFrame(ans)
#ans.to_csv('111/xgboost_result.csv', index = False)
#absolute_errors.to_csv('111/xgboost_error.csv', index = False)
plt.plot(absolute_errors, label='ABE', color='blue')
plt.show()
# 创建折线图
plt.plot(y_test, label='real', color='blue')  # 实际值
plt.plot(ans,  label='forecast', color='red', linestyle='--')  # 预测值
plt.legend(loc='upper right')
plt.title('real VS forecast')
plt.xlabel('date')
plt.ylabel('EGT')
plt.show()


# 评价指标
mae = mean_absolute_error(y_test, ans)
mse = mean_squared_error(y_test, ans)
rmse = sqrt(mse)
r2_score = r2_score(y_test, ans)

print('mae',mae)
print('mse',mse)
print('rmse',rmse)
print('r2_score',r2_score)
X_test = pd.DataFrame(X_test, columns=['spindle speed', 'feed speed', 'cutting width','cutting depth',\
                                       'overhang elongation'])
shap_values = shap.TreeExplainer(model).shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)
shap.dependence_plot('overhang elongation', shap_values, X_test, interaction_index='cutting width')
#shap.dependence_plot('overhang elongation', shap_values, X_test, interaction_index='spindle speed')
#shap.dependence_plot('overhang elongation', shap_values, X_test, interaction_index='feed speed')
#shap.dependence_plot('overhang elongation', shap_values, X_test, interaction_index='cutting depth')
shap.dependence_plot('cutting width', shap_values, X_test, interaction_index='spindle speed')
shap.dependence_plot('cutting width', shap_values, X_test, interaction_index='feed speed')
shap.dependence_plot('cutting width', shap_values, X_test, interaction_index='cutting depth')
shap.dependence_plot('overhang elongation', shap_values, X_test, interaction_index=None, show=False)
shap.dependence_plot('cutting width', shap_values, X_test, interaction_index=None, show=False)
# 显示重要特征
plot_importance(model)
plt.show()


