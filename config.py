# data

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
