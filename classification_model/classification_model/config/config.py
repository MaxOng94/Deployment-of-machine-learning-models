import pathlib

import classification_model
# data
PACKAGE_ROOT = pathlib.Path(classification_model.__file__).resolve().parent
# PACKAGE_ROOT is now the absolute file path to immediate parent of config -- > classification_model
TRAINING_DATA_FILE = "titanic.csv"
PIPELINE_NAME = "logistic_regression"




#============= features with missing values ==========

NUM_IMPUTE_VAR = ['age','fare']


#====== feature engineering variables ==========
FEATURE_ENGINEER = "cabin"



#================ variables =============

TARGET = "survived"

CATEGORICAL_VARIABLES = ['sex', 'cabin', 'embarked', 'title']

NUMERICAL_VARIABLES = ['pclass', 'age', 'sibsp', 'parch', 'fare']
