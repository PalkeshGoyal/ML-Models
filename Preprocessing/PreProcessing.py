import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
class PreprocessData():
    def __init__(self,df,target_col) -> None:
        self.df = df
        self.target_col = target_col
        self.encode_ref = {}
    
    def null_cols(self):
        null_cols = []
        for c_name in self.df.columns:
            count =  self.df[c_name].isnull().sum()
            if(count>0):
                null_cols.append(c_name)
        return null_cols


    def fill_null(self , cols = None):
        if(cols is None):
            cols = self.null_cols()
        for c_name in cols:
            if(self.df[c_name].dtype == "object"):
                self.df[c_name] = self.df[c_name].fillna(self.df[c_name].mode()[0])
            else:
                self.df[c_name] = self.df[c_name].fillna(self.df[c_name].median())

    def drop_columns(self,c_list):
        self.df =  self.df.drop(c_list,axis=1)
    
    def count_nulls(self):
        counts = 0
        for c_name in self.df.columns:
            na_count = self.df[c_name].isnull().sum()
            if(na_count>0):
                counts+=1
                print(f"{c_name} has {na_count} null values")
        if(counts == 0):
            print("No Null Values Present")
    def rename_columns(self,old,new):
        if(len(old) == len(new)):
            cols_dict = {}
            for old_name,new_name in zip(old,new):
                cols_dict[old_name] = new_name
            self.df.rename(columns = cols_dict, inplace = True)

    def get_categorical_cols(self,threshold = 0.01, get_col_perc = False):
        categorical_cols = []
        for c_name in self.df.columns:
            length = len(self.df[c_name].value_counts())
            perc = (length/self.df.shape[0])*100
            if(get_col_perc):
                print(f"{c_name} has {perc} of different values\n")
            if(perc<threshold):
                categorical_cols.append(c_name)
        return categorical_cols

    def encode_cols(self , cols = None, threshold = 0.01, get_col_perc = False):
        if(cols is None):
            cols = self.get_categorical_cols(threshold=threshold, get_col_perc= get_col_perc)
        for c_name in cols:
            self.encode_ref[c_name] = self.df.groupby(c_name)[self.target_col].agg("mean")
            self.df[c_name] = self.df.groupby(c_name)[self.target_col].transform("mean")
    
    def get_object_cols(self):
        dtypes = self.df.dtypes
        cols = self.df.columns
        obj_list = []
        for c_name , dtype in zip(cols,dtypes):
            if(dtype == "object"):
                obj_list.append(c_name)
        return obj_list 
    
    def iqr_treatment(self,df):
        Q3 = np.quantile(df, 0.75)
        Q1 = np.quantile(df, 0.25)
        IQR = Q3 - Q1
        lower_range = Q1 - 1.5 * IQR
        upper_range = Q3 + 1.5 * IQR
        df[df>upper_range] = upper_range
        df[df<lower_range] = lower_range
        return df

    def treat_outlier(self , cols = None):
        if(cols is None):
            cols = self.df.columns
            non_outlier_cols = [self.target_col]
            non_outlier_cols.extend(self.get_categorical_cols())
            for c_name in non_outlier_cols:
                if(c_name in cols):
                    cols =cols.drop(c_name)
        for c_name in cols:
            self.df[c_name] = self.iqr_treatment(self.df[c_name])

    def boxplot(self , cols = None):
        if(cols is None):
            cols = self.df.columns
            cols = cols.drop(self.target_col)
        fig , ax = plt.subplots( math.ceil(len(cols)/3) , 3)
        index = 0
        for axes in ax:
            for axx in axes:
                if(index<len(cols)):
                    sns.boxplot(self.df[cols[index]], ax=axx)
                    index+=1