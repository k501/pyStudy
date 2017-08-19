import numpy as np
import scipy as sp
import xgboost as xgb
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

iris = datasets.load_iris()
trainX = iris.data[0::2,:]
trainY = iris.target[0::2]
testX = iris.data[1::2,:]
testY = iris.target[1::2]

np.random.seed(131)

# Grid Search
params={'max_depth': [5],
        'subsample': [0.95],
        'colsample_bytree': [1.0]
}

xgb_model = xgb.XGBClassifier()
gs = GridSearchCV(xgb_model,
                  params,
                  cv=10,
                  scoring="log_loss",
                  n_jobs=1,
                  verbose=2)
gs.fit(trainX,trainY)
predict = gs.predict(testX)

print confusion_matrix(testY, predict)

# RandomizedSearchCV
param_distributions={'max_depth': sp.stats.randint(1,11),
                     'subsample': sp.stats.uniform(0.5,0.5),
                     'colsample_bytree': sp.stats.uniform(0.5,0.5)
}

xgb_model = xgb.XGBClassifier()
rs = RandomizedSearchCV(xgb_model,
                        param_distributions,
                        cv=10,
                        n_iter=20,
                        scoring="log_loss",
                        n_jobs=1,
                        verbose=2)
rs.fit(trainX,trainY)
predict = rs.predict(testX)

print confusion_matrix(testY, predict)