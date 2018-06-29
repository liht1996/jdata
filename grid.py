# Author: liht1996

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import classification_report
from model import feat_arange, score
import lightgbm as lgb

tuned_para_S1={
     'task': ['train'],
     'boosting_type': ['rf'],
     'objective': ['regression'],
     'metric': [{'l2'}],
     'max_depth':[5,6],
     'learning_rate': [0.06,0.05,0.04],
     'feature_fraction': [0.5,0.75,0.9],
     'bagging_fraction': [0.4,0.6,0.8,1.0],
     'bagging_freq': [5],
     'verbose': [0]
      }
tuned_para_S2={
     'task': ['train'],
     'boosting_type': ['rf'],
     'objective': ['regression'],
     'metric': [{'l2'}],
     'max_depth':[5,6],
     'learning_rate': [0.06,0.05,0.04],
     'feature_fraction': [0.5,0.75,0.9],
     'bagging_fraction': [0.4,0.6,0.8,1.0],
     'bagging_freq': [5],
     'verbose': [0]
      }

#tuned_para_S1={
#    'task': ['train'],
#    'boosting_type': ['gbdt'],
#    'objective': ['regression'],
#    'metric': [{'l2'}],
#    'num_leaves': [40,41],
#    'learning_rate': [0.05],
#    'feature_fraction': [0.9],
#    'bagging_fraction': [0.8],
#    'bagging_freq': [5],
#    'verbose': [0]
#     }
#tuned_para_S2={
#    'task': ['train'],
#    'boosting_type': ['gbdt'],
#    'objective': ['regression'],
#    'metric': [{'l2'}],
#    'num_leaves': [39,40],
#    'learning_rate': [0.05],
#    'feature_fraction': [0.9],
#    'bagging_fraction': [0.8],
#    'bagging_freq': [5],
#    'verbose': [0]
#     }


def gridSearch(X, y, tuned_parameters, test, score, testApril=True):
    
    if test=="S1":
        if testApril==False:
            X_train, X_test, y_train, y_test = train_test_split(X, y[::,0], test_size=0.3, random_state=0)
        else:
            X_test = X[0:99446]
            y_test = y[0:99446,0]
            X_train = X[99446::]
            y_train = y[99446::,0]

    elif test=="S2":
        if testApril==False:
            X_train, X_test, y_train, y_test = train_test_split(X, y[::,1], test_size=0.3, random_state=0)
        else:
            X_test = X[0:99446]
            y_test = y[0:99446,1]
            X_train = X[99446::]
            y_train = y[99446::,1]


    print("# Tuning hyper-parameters for %s" % score)
    print()

    LGB=lgb.LGBMRegressor()

    # 调用 GridSearchCV，将 LGB(), tuned_parameters, cv=5, 还有 scoring 传递进去
    gbm = GridSearchCV(LGB, tuned_parameters, scoring=score, verbose=1)

    # 用训练集训练这个学习器 gbm
    gbm.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()

    # 调用 gbm.best_params_ 就能直接得到最好的参数搭配结果
    print(gbm.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = gbm.cv_results_['mean_test_score']
    stds = gbm.cv_results_['std_test_score']

    # 看一下具体的参数间不同数值的组合后得到的分数是多少
    for mean, std, params in zip(means, stds, gbm.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        
    y_pred = gbm.predict(X_test)

    return y_pred, y_test






if __name__ == '__main__':
    filename='3609Nopa23allNoageOarea'

    test_score='explained_variance' 

    X, y, X_predict, fea_col = feat_arange(0,filename)

#    y_pred_S1, y_test_S1 = gridSearch(X, y, tuned_para_S1, "S1", test_score)

    y_pred_S2, y_test_S2 = gridSearch(X, y, tuned_para_S2, "S2", test_score)

#    score(y_pred_S1,y_pred_S2,y_test_S1,y_test_S2)
