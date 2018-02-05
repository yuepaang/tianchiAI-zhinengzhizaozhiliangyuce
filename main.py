import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from datetime import datetime

# 673 features
features = pd.read_csv('./data/feature.csv')
train_df = pd.read_excel('./data/train.xlsx')
test_df = pd.read_excel('./data/testB.xlsx')

namesIdx = features.iloc[:,1].values.tolist()
for i, _ in enumerate(namesIdx):
    namesIdx[i] = namesIdx[i].replace('X', '', 1)

z0 = train_df.loc[:,namesIdx]
z1 = test_df.loc[:,namesIdx]

# remove the date and time columns
z0.drop(z0.columns[np.where(z0.min()>1e13)], axis=1, inplace=True)
z1.drop(z1.columns[np.where(z1.min()>1e13)], axis=1, inplace=True)
# remove the NAs
z0.drop(z0.columns[209], axis=1, inplace=True)
z1.drop(z1.columns[209], axis=1, inplace=True)

z0['Y'] = train_df.Y
y0 = z0.pop('Y')

# XGBoost Model0
model_xgb0 = xgb.XGBRegressor(max_depth=10, learning_rate=0.1)
model_xgb0.fit(z0, y0)
pred13 = model_xgb0.predict(z1)

# XGBoost Model1
model_xgb1 = xgb.XGBRegressor(max_depth=9, learning_rate=0.1, n_estimators=120)
model_xgb1.fit(z0, y0)
pred_lu = model_xgb1.predict(z1)
pred13 = (pred13 + pred_lu) / 2

# XGBoost Model 2
model_xgb2 = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate =0.1,
    max_depth=7,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.7,
    colsample_bytree=0.7,
    nthread=7,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=2018)
model_xgb2.fit(z0, y0)
train_new_feature1 = model_xgb2.apply(z0)
test_new_feature1 = model_xgb2.apply(z1)
model_xgb2.fit(train_new_feature1, y0)
y_new_feature2= model_xgb2.predict(test_new_feature1)

pred14 = (pred13 * 0.90 + y_new_feature2 * 0.1)

# 173 features
with open('./data/x_train', 'rb') as f:
    x_train = pickle.load(f)

with open('./data/y_train', 'rb') as f:
    y_train = pickle.load(f)

with open('./data/x_testB', 'rb') as f:
    x_test = pickle.load(f)

# XGBoost Model 3
model_xgb3 = xgb.XGBRegressor(max_depth=7, learning_rate=0.1)
model_xgb3.fit(x_train, y_train)
pred_tmp = model_xgb3.predict(x_test)
pred16 = 0.8 * pred14 + 0.2 * pred_tmp

# pred_best = pd.read_csv('../Apred16_best.csv')
# print(mean_squared_error(pred16, pred_best.pred))

dt = datetime.now()
output = dt.strftime('%y%m%d_%H%M.csv')
df = pd.DataFrame({'pred': pred16.tolist()})
df.to_csv("{0}".format(output))