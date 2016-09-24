import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.linear_model import LassoCV
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression, Ridge

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_data = pd.concat((train.loc[:, 'Park_ID':'Location_Type'],
                      test.loc[:, 'Park_ID':'Location_Type']))
all_data['Day'] = all_data.Date.str.split('-').str[0].astype(int)
all_data['Month'] = all_data.Date.str.split('-').str[1].astype(int)
all_data['Year'] = all_data.Date.str.split('-').str[2].astype(int)
all_data = all_data.drop('Date', axis = 1)

train['Day'] = train.Date.str.split('-').str[0].astype(int)
train['Month'] = train.Date.str.split('-').str[1].astype(int)
train['Year'] = train.Date.str.split('-').str[2].astype(int)
train = train.drop('Date', axis = 1)

'''
li = all_data.Date.str.split('-').tolist()
for i, j in enumerate(li):
    li[i] = int("".join(reversed(j)))
all_data['Date'] = pd.Series(li)

li = train.Date.str.split('-').tolist()
for i, j in enumerate(li):
    li[i] = int("".join(reversed(j)))
train['Date'] = pd.Series(li)
'''
# For right skewed data
skewed_feats = train.apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.8]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# For left skewed data
skewed_feats_left = train.apply(lambda x: skew(x.dropna()))
skewed_feats_left = skewed_feats_left[skewed_feats_left < -0.8]
skewed_feats_left = skewed_feats_left.index

all_data[skewed_feats_left] = np.power(all_data[skewed_feats_left], 3)

all_data = all_data.fillna(all_data.median())
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.Footfall
'''
model = LassoCV(alphas=[1, 0.1, 0.001, 0.0005], max_iter = 2500)
model.fit(X_train, y)
preds = model.predict(X_test)

model = LinearRegression()
model.fit(X_train, y)
preds = model.predict(X_test)

model = Ridge(10)
model.fit(X_train, y)
preds = model.predict(X_test)

model = RandomForestRegressor()
model.fit(X_train, y)
preds = model.predict(X_test)
'''
model = GradientBoostingRegressor()
model.fit(X_train, y)
preds = model.predict(X_test)

solution = pd.DataFrame({"ID": test.ID, "Footfall": preds})
solution.to_csv("Try2.csv", index=False)
