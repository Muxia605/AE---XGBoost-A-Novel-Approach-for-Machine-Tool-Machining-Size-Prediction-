import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedKFold, KFold
from glob import glob
import numpy as np
import os
import smogn
import resreg
from xgboost import XGBRegressor
import itertools as it

from imbalance_metrics import regression_metrics as rm
import ImbalancedLearningRegression as iblr

import warnings
warnings.filterwarnings('ignore')

def train(regressor, strategy, X, y, c, dataset_name,fold):

  train = np.column_stack((y, X))
  train_output_file = f"name_{dataset_name}.csv"
  pd.DataFrame(train).to_csv(train_output_file, index=False)
  train = pd.read_csv(train_output_file)

  try:
      train = balance(train, strategy, c)
  except ValueError:
      pass
  balanced_train_output_file = f"balanced_train_{strategy}_{dataset_name}_fold{fold}.csv"
  train.to_csv(balanced_train_output_file, index=False)

  X = train.drop([train.columns[0]], axis=1)
  y = train[train.columns[0]]

  model = regressor.fit(X.values, y.values)

  return model

def balance(train, strategy, c):

  if strategy == "SG":
    train =  train.dropna()
    train = smogn.smoter(data = train, y = train.columns[0], samp_method=c[2], k=c[0], pert=c[1], rel_xtrm_type = 'high', rel_thres = 0.8)
    train =  train.dropna()
  elif strategy == "SMT":
    train = iblr.smote(data = train, y = "0", samp_method=c[0], rel_thres = 0.8)
  return train

def repeatedKfold(X, y, dataset_name):

  outer = RepeatedKFold(n_splits=10, n_repeats=2, random_state=42)
  inner = KFold(n_splits=2, random_state=42, shuffle=True)

  print(outer)

  all_result = []

  v_pert = np.arange(0, 1.05, 0.05).tolist()
  val = np.arange(0.3, 1, 0.2).tolist()

  strategys = {"SMT":{"C.perc":["balance", "extreme"], "k": [3, 5, 7]},
               "SG":{"samp_method":["balance", "extreme"], "k": [3, 5, 7], "pert": v_pert},
               }

  regressors = {
    'XG': XGBRegressor()
  }

  all_results_df = pd.DataFrame(columns=['Fold', 'Strategy', 'BestC', 'BestSERA'])

  for strategy in strategys:
      print(strategy)
      data_frame = []
      params = strategys[strategy]
      keys = sorted(params)

      for regressor_name, regressor in regressors.items():
        print(regressor_name)
        for fold, (train_index, test_index) in enumerate(outer.split(X, y)):
            print("outer")
            print("Fold:", fold)
            X_train_outer, X_test_outer = X[train_index], X[test_index]
            y_train_outer, y_test_outer = y[train_index], y[test_index]

            best_sera = float('inf')
            best_c = None


            combinations = it.product(*(params[Name] for Name in keys))

            for c in combinations:
                print(strategy)
                print(c)

                fold_scores = []

                for train_inner_index, val_inner_index in inner.split(X_train_outer, y_train_outer):
                    score_perc = []
                    print("Inner loop")

                    X_train_inner, X_val_inner = X[train_inner_index], X[val_inner_index]
                    y_train_inner, y_val_inner = y[train_inner_index], y[val_inner_index]

                    model_inner = train(regressor, strategy, X_train_inner, y_train_inner, c, dataset_name, fold)
                    y_pred = model_inner.predict(X_val_inner)
                    y_val_inner = y_val_inner.flatten()
                    y_pred = y_pred.flatten()
                    sera = rm.sera(y_val_inner, y_pred)
                    fold_scores.append(sera)

                avg_sera = np.mean(fold_scores)
                print("Average SERA:", avg_sera)

                if avg_sera < best_sera:
                    best_sera = avg_sera
                    best_c = c
                print(best_c)

            model_outer = train(regressor, strategy, X_train_outer, y_train_outer, best_c, dataset_name, fold)
            y_pred_outer = model_outer.predict(X_test_outer)
            y_test_outer = y_test_outer.flatten()
            y_pred_outer = y_pred_outer.flatten()
            sera_outer = rm.sera(y_test_outer, y_pred_outer)
            print("sera_outer", sera_outer)

            model_name = type(model_outer).__name__

            #test = np.column_stack((test_index, y_test_outer))
            #pd.DataFrame(test).to_csv('Content/'+strategy+'/'+dataset_name+'.csv/'+model_name+'/Test{}_{}_{}.csv'.format(fold, strategy, model_name, index = False))
            pred = np.column_stack((test_index, y_pred_outer))
            pd.DataFrame(pred).to_csv(r'your_path\Pred{}_{}_{}.csv'.format(fold, strategy, model_name, index = False))
dataset_path = 'your_dataset.csv'
dataset_name = dataset_path.split('/')[-1].replace('.csv', '')
X = pd.DataFrame(pd.read_csv("your_train_x.csv")).values
y = pd.DataFrame(pd.read_csv("your_train_y.csv")).values
repeatedKfold(X, y, dataset_name)