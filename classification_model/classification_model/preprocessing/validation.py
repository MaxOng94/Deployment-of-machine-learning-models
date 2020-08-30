from classification_model.config import config
import pandas as pd

# bascially, a function that removes rows that contains na.
# this function is for prediction, so we are going to put it in prediction
def validate_inputs(input_data :pd.DataFrame) -> pd.DataFrame:
    validated_data = input_data.copy()
    if validated_data[config.NUMERICAL_VARIABLES].isnull().any().any():
        validated_data= validated_data.dropna(axis = 0, subset =config.NUMERICAL_VARIABLES)

    if validated_data[config.CATEGORICAL_VARIABLES].isnull().any().any():
        validated_data.dropna(axis = 0, subset =config.CATEGORICAL_VARIABLES)

    return validated_data
