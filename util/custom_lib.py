"""
    custom library for data science
    author: yeounoh chung (yeounohster@gmail.com)
    date: 4/10/18

    - Pipeliner: Automatic ML pipeline tuning
    - DataWrangler: Data ingestion / pre-processing
    - VisualAnalyzer: Plotting
"""
import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from time import time
import concurrent.futures
import sys, os
if 'util' in os.getcwd(): from optimizer import *
else: from util.optimizer import * 

from sklearn.svm import SVC, SVR  # fit time is quadratic with the n_example (also, one-vs-one multiclass support)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

import matplotlib.pyplot as plt

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')
    
class DataWrangler:
    """
        Data ingestion / pre-processing
    """

    def __init__(self):
        pass

    def read_csv_to_df(self, src_path, timestamp=None):
        df = pd.read_csv(src_path)

        # timestamp tuple, e.g., ('Acitivity Period', '%Y%m')
        # pd.to_datetime(df[ ... ], format='%Y%m', errors='ignore')
        if timestamp is not None:
            df[timestamp[0]] = pd.to_datetime(df[timestamp[0]],
                                              format=timestamp[1],
                                              errors='ignore')
        # timestamp, if not specified, will be parsed as number

        # asume all categorical are of type 'object'
        cat_cols = list(df.dtypes[df.dtypes == 'category'].index)
        df[cat_cols] = df[cat_cols].astype('object')
        return df

    def drop_id_columns(self, df):
        drop_cols = []
        for col in list(df.columns):
            n_uniq = df[col].value_counts().shape[0]
            if n_uniq == df.shape[0]:
                drop_cols.append(col)
        return df.drop(drop_cols, axis=1)

    def stack_dfs(self, dfs):
        ''' Stack train and test data before any transformation '''
        stacked_df = pd.concat(dfs)
        splits = [len(df) for df in dfs[:-1]]
        return stacked_df, splits

    def scale_numeric_columns(self, df): 
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        num_cols = list(df.dtypes[df.dtypes != 'object'].index)
        df[num_cols] = scaler.fit_transform(df[num_cols].as_matrix())

        return df, scaler

    def encode_categorical_columns(self, df_train, df_test=None): 
        if df_test is None:
            df = df_train
        else:
            df = pd.concat([df_train, df_test])

        cat_cols = list(df.dtypes[df.dtypes == 'object'].index)
        df = pd.get_dummies(df, columns = obj_cols)

        return df[:len(df_train)], df[len(df_train):]

    def label_imbalance_check_fix(self, df, target, verbose=False):
        ''' applicable for classfiication problem '''
        print('Checking label imbalanced-ness')
        ratios = df[target].value_counts() / df.shape[0]
        if verbose:
            #print(df[target].value_counts())
            print(ratios)

        if ratios.max() - ratios.min() > 0.5:
            r_min = ratios.min()
            dfs = []
            for l, r in list(zip(ratios.index, ratios.values)):            
                dfs.append(df[df[target] == l].sample(frac=r_min/r))
            df = pd.concat(dfs)  
        return df

    def missing_value_check_fix(self, df, verbose=False):
        print('Checking missing value by attribute:')
        missing_prop = df.isnull().sum().div(df.count())        
        if verbose:
            print(list(zip(missing_prop[missing_prop > 0.].index, 
                           missing_prop[missing_prop > 0.])))

        for col in list(df.columns):
            if missing_prop[col] > 0.2:
                ''' drop column if missign too many '''
                df = df.drop([col], axis=1)
            elif missing_prop[col] > 0.:
                if df.dtypes[col] == 'object':
                    ''' common value imputation for categorical '''
                    most_common_val = df[col].value_counts().index[0]
                    df = df.fillna({col: most_common_val})
                else:
                    ''' average value imputation for numeric '''
                    avg_val = df.mean()[col]
                    df = df.fillna({col: avg_val})

    def reduce_dimensions(self, df_train, df_test):
        from sklearn.decomposition import PCA
        pca = PCA(.99)
        x_train = pca.fit_transform(df_train.as_matrix())
        x_test = pca.transform(df_test.as_matrix())
        df_train_ = pd.DataFrame(data=x_train,index=df_train.index)
        df_test_ = pd.DataFrame(data=x_test,index=df_test.index)

        return df_train_, df_test_

lr_params = {"C": [1, 1e3, 1e5], "solver": ['lbfgs']}
svm_params = {"C": [1000, 10, 1, 1e3]} 
rf_param_dist = {"max_depth": [None],
                 "n_estimators": [10, 100, 200, 500, 1000],
                 "max_features": [None, 0.2, 0.3, 0.5], #None defaults to 'auto'
                 "min_samples_split": randint(2, 101),
                 "min_samples_leaf": randint(1, 500),
                 "bootstrap": [True],
                 "criterion": ["gini", "entropy"],
                 "n_jobs": [-1]}
xgb_param_dist = {"max_depth": randint(3,11),
                  "learning_rate": [0.2, 0.1, 0.03, 0.01],
                  "n_estimators": [10, 100, 200, 500, 1000],
                  "n_jobs": [-1],
                  "gamma": randint(0,11),
                  "min_child_weight": randint(1,11),
                  "subsample": [0.5, 1],
                  "colsample_bylevel": [0.5, 1],
                  "reg_lambda": randint(1,100)
                  }

class Pipeliner:

    def __init__(self):
        # pre-defined search space
        self.model_space = []
        self.model_space.append((LogisticRegression(), lr_params))
        self.model_space.append((SVC(), svm_params)) 
        self.model_space.append((RandomForestClassifier(), rf_param_dist))
        self.model_space.append((XGBClassifier(), xgb_param_dist))  

    def curate_pipeline(self, X, y, budget=50, cv=3, scorer=accuracy_score):
        #rs = RandomOptimizer(self.model_space, budget=budget)
        rs_bandit = GreedyOptimizer(self.model_space, budget=budget)

        start = time()
        #rs = rs.parallel_fit(X, y, cv=cv, scorer=scorer)
        rs_bandit = rs_bandit.fit(X, y, cv=cv, scorer=scorer)
        end = time()
        print ('RandomBanditOptimizer took %s seconds'%(end-start))
        print ('best score: ', rs.best_score_)
        print ('using best model: ', rs.best_estimator_)

        return rs.best_estimator_ 

    def random_forest_search(self, X, y, budget=20, cv=3, scoring='f1_weighted'):
        ''' curate a random forest pipeline '''
        from sklearn.model_selection import RandomizedSearchCV
        rs = RandomizedSearchCV( RandomForestClassifier(), rf_param_dist, 
                                n_iter=budget, cv=cv, n_jobs=-1, scoring=scoring)
        start = time()
        rs = rs.fit(X, y)
        end = time()
        print ('RandomOptimizer took %s seconds'%(end-start))
        print ('best score: ', rs.best_score_)
        print ('using best model: ', rs.best_estimator_)

        return rs.best_estimator_

    def XGB_search(self, X, y, budget=20, cv=3, scoring='f1_weighted'):
        pass

class VisualAnalyzer:

    def __init__(self):
        pass

    def plot_learning_curve(self, model, X, y, cv=5, 
                            train_sizes=np.linspace(.1,1.0,5),
                            title=''):
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        
        plt.figure()
        plt.title(title)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
        plt.legend(loc="best")

        return plt
        
    def plt_confusion_matrix(self, y_test, y_preds):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_preds)

        return cm


if __name__ == '__main__':
    #from sklearn.datasets import fetch_mldata
    #mnist = fetch_mldata('MNIST original')
    from sklearn.datasets import load_digits
    mnist = load_digits()

    x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2)
    tuner = Pipeliner()
    model = tuner.curate_pipeline(x_train, y_train)
    print(model.score(x_test, y_test))
