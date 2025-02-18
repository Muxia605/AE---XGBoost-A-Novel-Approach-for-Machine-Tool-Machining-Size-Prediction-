import xgboost as xgb
import smogn
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

# # Parameter optimization
# cv_params = {'max_depth': [5, 6, 7, 9], 'gamma':[0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1],\
#  'learning_rate': [0.01, 0.015, 0.025, 0.05], 'subsample':[0.6, 0.7, 0.8, 0.9, 1],\
#              'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1]}
# other_params = {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 400, 'gamma': 0,'min_child_weight': 1,
#                 'subsample': 1, 'colsample_bytree': 0.8, 'reg_alpha': 0.05, 'reg_lambda': 0.1, 'seed': 0}
# model = xgb.XGBRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
# optimized_GBM.fit(X, y)
# evalute_result = optimized_GBM.cv_results_
# print('The results in each iteration：{0}'.format(evalute_result))
# print('Best_params：{0}'.format(optimized_GBM.best_params_))
# print('Best_score:{0}'.format(optimized_GBM.best_score_))

model = xgb.XGBRegressor(max_depth=6,
                        learning_rate=0.05,
                        n_estimators=400,
                        objective='reg:squarederror',
                        booster='gbtree',
                        gamma=0,
                        min_child_weight=1,
                        subsample=1,
                        colsample_bytree=0.8,
                        reg_alpha=0.05,
                        reg_lambda=0.1,
                        random_state=0)
model.fit(X,y)

# Prediction
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

# Plot
# Evaluation metrics
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

plot_importance(model)
plt.show()


