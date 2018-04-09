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
        stacked_df = pd.concat(dfs)
        splits = [len(df) for df in dfs[:-1]]
        return stacked_df, splits

    def encode_columns(self, df): 
        from sklearn.preprocessing import StandardScaler
        num_cols = list(df.dtypes[df.dtypes != 'object'].index)
        df[num_cols] = StandardScaler().fit_transform(df[num_cols])

        cat_cols = list(df.dtypes[df.dtypes == 'object'].index)
        df = pd.get_dummies(df, columns = obj_cols)
        return df

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

        return df

class Pipeliner:
    
    def __init__(self):
        pass


class VisualAnalyzer:

    def __init__(self):
        pass
