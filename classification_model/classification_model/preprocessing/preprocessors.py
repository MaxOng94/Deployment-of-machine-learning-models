import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
# this will contain all the classes for transformers

# we will create a class IMPUTE NUM
# this class will derive the mode from the given dataset and variables given that needs
# to be imputed
class MissingImputer(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        # this is to check if the variables passed to the object is a list of variables name
        # likely will be, since we will call the list of variables from config file
        # if variables passed not list but string, change to list, else self.variables = var
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    # input to be a dataframe
    def fit(self, X, y = None):
        return self

    def transform(self, X):
        X = X.copy()
        for num_var in self.variables:
            #create two additional columns, binary missing indicator
            X[num_var + "_NA"] = np.where(X[num_var].isnull(),1,0)
        return X


class num_imputer(BaseEstimator, TransformerMixin):
    """
    this class takes a list of variables that has empty rows and fit with their own mode

    """
    def __init__(self, variables=None):
        # this is to check if the variables passed to the object is a list of variables name
        # likely will be, since we will call the list of variables from config file
        # if variables passed not list but string, change to list, else self.variables = var
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    # input to be a dataframe
    def fit(self, X, y = None):
        # CREATE an empty dict to store the num_vars with empty rows
        # and also their mode value
        # fitted NUM_IMPUTE_DICT
        self.NUM_IMPUTE_DICT = {}
        for num_var in self.variables:
            self.NUM_IMPUTE_DICT[num_var] = X[num_var].median()
        return self

    def transform(self, X):
        X = X.copy()

        for num_var in self.variables:
            #create two additional columns, binary missing indicator

            # fill in the empty rows with the mode values from the fitted NUM_IMPUTE_DICT
            X[num_var].fillna(self.NUM_IMPUTE_DICT[num_var],inplace = True)
        return X

# =========below are all classes for categorical variables ======================

class cat_feature_engineer(BaseEstimator, TransformerMixin):
    """
    this class takes in a list of variables and extract only the first letter and drops the rest of the variable value
    """
    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y = None):
        # needs to have this regardless to be able to be placed in the pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for cat_var in self.variables:
            X[cat_var] = X[cat_var].str[0]

        return X


class cat_imputer(BaseEstimator, TransformerMixin):
    """
    this class takes in a list of categorical variables and fill in the missing rows with "Missing"
    """

    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables


    def fit(self, X, y = None):
        # is there anything to fit on the data for our transformer?
        return self

    def transform(self, X):
        X = X.copy()
        for cat_var in self.variables:
            X[cat_var].fillna("Missing",inplace = True)
        return X

class remove_rare_labels(BaseEstimator,TransformerMixin):
    """
    this class takes in categorical variables and find the most frequent labels in them.

    """
    def __init__(self,variables = None,percent = 0.05):
        self.percent = percent
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables


    def fit(self, X, y = None):
        X = X.copy()
        self.frequent_label_dict = {}
        # Use dictionary to represent sex : ["male", "female"]
        for cat_var in self.variables:
            # get the label
            tmp = X[cat_var].value_counts(normalize = True)
            self.frequent_label_dict[cat_var]= list(tmp[tmp > self.percent].index)
        return self

    def transform(self, X):
        X = X.copy()
        for cat_var in self.variables:
            X[cat_var] = np.where(X[cat_var].isin(self.frequent_label_dict[cat_var]),X[cat_var],"Rare")
        return X


class one_hot_encode(BaseEstimator, TransformerMixin):
    """One hot encodes categorical variables
    """

    def __init__(self, variables = None):
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X, y = None):
        self.dummies = pd.get_dummies(X[self.variables],drop_first = True).columns
        # is there anything to fit on the data for our transformer?
        return self


    def transform(self, X):
        X = X.copy()
        X = pd.concat([X,pd.get_dummies(X[self.variables],drop_first = True)], axis = 1)
        X.drop(labels = self.variables, axis = 1, inplace = True)
        # add missing dummies if any
        missing_vars = [var for var in self.dummies if var not in X.columns]

        if len(missing_vars) != 0:
            for var in missing_vars:
                X[var] = 0
        return X
