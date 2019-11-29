import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, make_scorer

from sklearn.linear_model import SGDRegressor, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier

from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD


# Plot Feature Importance -------------------------------------------------------------------------------------------------------------------
def plot_feature_importance(feature_importance, folds):
    
    feature_importance["Importance"] /= folds.n_splits
    cols = feature_importance[["Feature", "Importance"]].groupby("Feature").mean().sort_values(
            by="Importance", ascending=False)[:50].index

    best_features = feature_importance.loc[feature_importance.Feature.isin(cols)]

    plt.figure(figsize=(16, 12))
    sns.barplot(x="Importance", y="Feature", data=best_features.sort_values(by="Importance", ascending=False))
    plt.title('LGB Features (avg over folds)');
    pass

# Regression -------------------------------------------------------------------------------------------------------------------
# LGB -------------------------------------------------------------------------------------------------------------------
def my_lgb(X, X_test, y, params, folds):
    result_dict = {}
    cols = X.columns
    oof = np.zeros(len(X)) # Out-of-fold
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()

    for fold_, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('\n-------------------------------------------------------------------------------------\n')
        print(f'Fold {fold_ + 1} started at {time.ctime()}')

        # split training set and validation set
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        # start modeling
        model = lgb.LGBMRegressor(**params, n_jobs = -1)
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_metric="mse",
                  verbose = 2000,
                  early_stopping_rounds = 1200)
        
        # prediction ----------------------------------------------------------------------
        # y prediction from the model
        y_pred_valid = model.predict(X_valid)
        # record y prediction on validation index (out-of-fold)
        oof[valid_index] =  y_pred_valid.reshape(-1,)
        
        # y prediction for submission
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
        # record y prediction for submission
        prediction       += y_pred
        
        # feature importance ---------------------------------------------------------------
        fold_importance = pd.DataFrame()
        fold_importance["Feature"] = cols
        fold_importance["Importance"] = model.feature_importances_
        fold_importance["Fold"] = fold_ + 1
        
        # concatenate feature importance dataframe
        feature_importance = pd.concat([feature_importance, fold_importance], axis = 0)

        # score -----------------------------------------------------------------------------
        
        #score = r2_score(y_valid, y_pred_valid)
        score = np.sqrt(mean_squared_error(np.expm1(y_valid), np.expm1(y_pred_valid)))
        scores.append(score)
        print("score: ", score)
            
    prediction /= folds.n_splits
    
    # change here depending on your problem
    oof = np.expm1(oof)
    prediction = np.expm1(prediction)
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    return {'result_dict':result_dict, 'feature_importance':feature_importance, 'score':np.mean(scores)}

# XGB -------------------------------------------------------------------------------------------------------------------
def my_xgb(X, X_test, y, params, folds):
    result_dict = {}
    cols = X.columns
    oof = np.zeros(len(X)) # Out-of-fold
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()

    for fold_, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('\n-------------------------------------------------------------------------------------\n')
        print(f'Fold {fold_ + 1} started at {time.ctime()}')

        # split training set and validation set
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=cols)
        valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=cols)


        # start modeling
        watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
        model = xgb.train(dtrain=train_data,
                          params=params,
                          num_boost_round=1000,
                          evals=watchlist,
                          early_stopping_rounds=50,
                          verbose_eval=100)
        
        # prediction ----------------------------------------------------------------------
        # y prediction from the model
        y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names = cols), ntree_limit=model.best_ntree_limit)
        # record y prediction on validation index (out-of-fold)
        oof[valid_index] =  y_pred_valid.reshape(-1,)
        
        # y prediction for submission
        y_pred = model.predict(xgb.DMatrix(X_test, feature_names = cols), ntree_limit=model.best_ntree_limit)
        # record y prediction for submission
        prediction       += y_pred
        
        # feature importance ---------------------------------------------------------------
        fi = model.get_score(importance_type='gain')
        fold_importance = pd.DataFrame()
        fold_importance["Feature"] = fi.keys()
        fold_importance["Importance"] = fi.values()
        fold_importance["Fold"] = fold_ + 1
        
        # concatenate feature importance dataframe
        feature_importance = pd.concat([feature_importance, fold_importance], axis = 0)

        # score -----------------------------------------------------------------------------
        
        #score = r2_score(y_valid, y_pred_valid)
        score = np.sqrt(mean_squared_error(np.expm1(y_valid), np.expm1(y_pred_valid)))
        scores.append(score)
        print("score: ", score)
            
    prediction /= folds.n_splits
    
    # change here depending on your problem
    oof = np.expm1(oof)
    prediction = np.expm1(prediction)
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    return {'result_dict':result_dict, 'feature_importance':feature_importance, 'score':np.mean(scores)}




def regression_1(X, X_test, y, params, folds, model_type):
    
    result_dict = {}
    cols = X.columns
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    
    for fold_, (train_index, valid_index) in enumerate(folds.split(X, y)):
        print('\n-------------------------------------------------------------------------------------\n')
        print(f'Fold {fold_ + 1} started at {time.ctime()}')
        
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        
        # ------------------------------------------------------------------------------------------
        if model_type=='cat':
            print("3")
            model = CatBoostRegressor(**params)
            model.fit(X_train,
                      y_train,
                      eval_set=(X_valid, y_valid), cat_features=[],
                      use_best_model=True,
                      verbose=False)
                
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        # ------------------------------------------------------------------------------------------
        if model_type=='gboost':
            model = GradientBoostingRegressor(**params)
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)

        # ------------------------------------------------------------------------------------------

        oof[valid_index] =  y_pred_valid.reshape(-1,)
        prediction       += y_pred
        score = r2_score(y_valid, y_pred_valid)
#        score = mean_squared_error(y_valid, y_pred_valid)
        scores.append(score)
        print("score: ", score)
        # scores.append(np.sqrt(mean_squared_error(y_valid, y_pred_valid)))
        
    prediction /= folds.n_splits
    # prediction = np.expm1(prediction) # !!!!!!!!!!!!!!!!!!!!!
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    return {'result_dict':result_dict, 'feature_importance':feature_importance, 'score':np.mean(scores)}
