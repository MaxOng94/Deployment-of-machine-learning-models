# this will contain the main pipeline
from classification_model.preprocessing import preprocessors as pf
from classification_model.config import config
#==================================
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



pipe = Pipeline(
[
    ("cat_feature_engineer",
        pf.cat_feature_engineer(variables =config.FEATURE_ENGINEER)),

    ("cat_imputer",
        pf.cat_imputer(variables = config.CATEGORICAL_VARIABLES)),

    ("MissingImputer",
        pf.MissingImputer(variables = config.NUM_IMPUTE_VAR)),

    ("num_imputer",
        pf.num_imputer(variables = config.NUM_IMPUTE_VAR)),

    ("remove_rare_labels",
        pf.remove_rare_labels(variables = config.CATEGORICAL_VARIABLES,percent = 0.05)),

    ("one_hot_encode",
        pf.one_hot_encode(variables = config.CATEGORICAL_VARIABLES)),

        ("scaler",StandardScaler()),

        ("log_reg", LogisticRegression(C=0.0005, random_state=0))



]
)
