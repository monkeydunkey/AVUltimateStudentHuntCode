import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.grid_search import GridSearchCV
import xgboost as xgb

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Coverting Location type to str to turn it into an object

train = train.drop('Location_Type', axis=1)
test = test.drop('Location_Type', axis=1)
# test['Location_Type'] = test['Location_Type'].astype(str)

all_data = pd.concat((train.loc[:, 'Park_ID':'Min_Moisture_In_Park'],
                      test.loc[:, 'Park_ID':'Min_Moisture_In_Park']))
# extracting month and day information from the date
all_data['Day'] = all_data.Date.str.split('-').str[0].astype(int)
all_data['Month'] = all_data.Date.str.split('-').str[1].astype(int)
all_data = all_data.drop('Date', axis=1)

train['Day'] = train.Date.str.split('-').str[0].astype(int)
train['Month'] = train.Date.str.split('-').str[1].astype(int)
train = train.drop('Date', axis=1)

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

# For right skewed data
skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.8]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# For left skewed data
skewed_feats_left = train[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats_left = skewed_feats_left[skewed_feats_left < -0.8]
skewed_feats_left = skewed_feats_left.index

all_data[skewed_feats_left] = np.power(all_data[skewed_feats_left], 3)


all_data = all_data.fillna(all_data.median())
'''
# Caculating the dummy variables - For location type
all_data = pd.get_dummies(all_data)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
'''

X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.Footfall


def rmse_cv(model):
    rmse = -cross_val_score(model, X_train, y, scoring="mean_squared_error",
                            cv=5)
    return rmse

model = GradientBoostingRegressor(n_estimators=800, min_samples_leaf=50,
                                  loss='huber', learning_rate=0.04,
                                  max_depth=14, max_features='sqrt',
                                  subsample=0.8, random_state=10)

model.fit(X_train, y)
preds = model.predict(X_test)

solution = pd.DataFrame({"ID": test.ID, "Footfall": np.round(preds)})
solution.to_csv("Try2.csv", index=False)
