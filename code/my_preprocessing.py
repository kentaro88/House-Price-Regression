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

from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD

"""
outlier
scaling
null value
feature engineering
encoding
dimentionality reduction
delete columns
"""

def outlier(df):
    print("Outlier ------------------------------------------------------")
    new_df = df.drop(df[(df['GrLivArea']>4000) & (df['SalePrice']<300000)].index)
    new_df = new_df.drop(new_df[new_df['LotArea']>100000].index)
    new_df = new_df.drop(new_df[new_df['LotFrontage']>300].index)
    new_df = new_df.drop(new_df[new_df['BsmtFinSF1']>5000].index)
    new_df = new_df.drop(new_df[new_df['TotalBsmtSF']>6000].index)
    new_df = new_df.drop(new_df[new_df['1stFlrSF']>4000].index)
    new_df = new_df.drop(new_df[(new_df['OpenPorchSF']>500) & (new_df['SalePrice']<100000)].index)
    return new_df

def feature_engineering(df):
    print("feature_engineering ------------------------------------------------------")
    begin = len(df.columns)
    # Merge Porch infomation
    df['TotalHousePorchSF'] = df['EnclosedPorch']+df['OpenPorchSF']+df['WoodDeckSF']+df['3SsnPorch']+df['ScreenPorch']

    # Total Floor Square Feet
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    # Squred Overall Quality because Overall Quality is the most correlated
    df['OverallQual_2'] = df['OverallQual']**2

    end = len(df.columns)
    print("{} cols are created".format(end-begin))
    return df

def null_values(df):
    print("null_values ------------------------------------------------------")
    print("beginning null values: {}".format(df.isna().sum().sum()))
    # categorical -------------------------------------------------
    df["MSZoning"] = df["MSZoning"].fillna("RL")
    df["Alley"] = df["Alley"].fillna("NONE")
    df = df.drop(['Utilities'], axis=1)
    df["Exterior1st"] = df["Exterior1st"].fillna("VinylSd")
    df["Exterior2nd"] = df["Exterior2nd"].fillna("VinylSd")
    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["BsmtQual"] = df["BsmtQual"].fillna("NONE")
    df["BsmtCond"] = df["BsmtCond"].fillna("NONE")
    df["BsmtExposure"] = df["BsmtExposure"].fillna("NONE")
    df["BsmtFinType1"] = df["BsmtFinType1"].fillna("NONE")
    df["BsmtFinType2"] = df["BsmtFinType2"].fillna("NONE")
    df["Electrical"] = df["Electrical"].fillna("SBrkr")
    df["KitchenQual"] = df["KitchenQual"].fillna("TA")
    df["Functional"] = df["Functional"].fillna("Typ")
    df["FireplaceQu"] = df["FireplaceQu"].fillna("NONE")
    df["GarageType"] = df["GarageType"].fillna("NONE")
    df["GarageFinish"] = df["GarageFinish"].fillna("NONE")
    df["GarageQual"] = df["GarageQual"].fillna("NONE")
    df["GarageCond"] = df["GarageCond"].fillna("NONE")
    df["PoolQC"] = df["PoolQC"].fillna("NONE")
    df["Fence"] = df["Fence"].fillna("NONE")
    df["MiscFeature"] = df["MiscFeature"].fillna("NONE")
    df["SaleType"] = df["SaleType"].fillna("WD")

    # numerical ----------------------------------------------------
    df["LotFrontage"] = df["LotFrontage"].fillna(np.mean(df["LotFrontage"]))
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["BsmtFinSF1"] = df["BsmtFinSF1"].fillna(0)
    df["BsmtFinSF2"] = df["BsmtFinSF2"].fillna(0)
    df["BsmtUnfSF"] = df["BsmtUnfSF"].fillna(0)
    df["TotalBsmtSF"] = df["TotalBsmtSF"].fillna(0)
    df["BsmtFullBath"] = df["BsmtFullBath"].fillna(0)
    df["BsmtHalfBath"] = df["BsmtHalfBath"].fillna(0)
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
    df["GarageCars"] = df["GarageCars"].fillna(0)
    df["GarageArea"] = df["GarageArea"].fillna(0)

    print("end null values: {}".format(df.isna().sum().sum()))
    return df

def columns_reduction(df):
    print("columns_reduction ------------------------------------------------------")
    # Multicolinearity -------------------------------------------------

    print("beginning ", len(df.columns)," columns")
    corr_matrix = df.corr().abs() # compute correration coefficient for all pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))     # get only one side of pair
    to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]     # get the columns whose correration coefficient exceeds to the threshold and drop them
    df = df.drop(to_drop, axis = 1)
    print("removed ", len(to_drop)," columns")
    print(to_drop)
    print("now ", len(df.columns)," columns")

    return df

def scaling(df):
    print("scaling ------------------------------------------------------")
    numericals = df.dtypes[df.dtypes != "object"].index
    scaler = StandardScaler()
    df[numericals] = scaler.fit_transform(df[numericals] )
    return df

def encoding(df, kind = "label"):
    print("encoding ------------------------------------------------------")
    print("beginning shape: ", df.shape)
    categoricals = df.dtypes[df.dtypes == "object"].index
    print(kind)

    if kind == "dummy":
        df = pd.get_dummies(df)
        
    elif kind == "onehot":
        pass

    elif kind == "label":
        for c in categoricals:
            lbl = LabelEncoder() 
            lbl.fit(list(df[c].values)) 
            df[c] = lbl.transform(list(df[c].values))
            
    print("categorical cols: ", len(df.dtypes[df.dtypes == "object"].index))
    print("beginning shape: ", df.shape)
    return df

def dimentionality_reduntion(df):
    return df
