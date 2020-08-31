from classification_model.config import config

from classification_model.preprocessing.data_management import load_data, load_pipeline
from classification_model.preprocessing.validation import validate_inputs


import joblib

import pandas as pd

# write a function that uses the load_pipeline (from data management) and return prediction using
# first line of titanic.csv

# X input to be X_train[0:1] --> this will give us a dataframe type
# output --> np.ndarray
def predict(*, input_data) ->pd.Series:
    # data becomes a dataframe
    pipe = load_pipeline(file_name = "logistic_regression0.1.0.pkl")
    validated_input= validate_inputs(input_data)
    y_pred = pipe.predict(validated_input)
    return y_pred



# def predict():
#     data = pd.read_csv(config.TRAINING_DATA_FILE)
#     X_train,X_test,y_train,y_test = train_test_split(data.drop(config.TARGET, axis=1),  # predictors
#                                    data[config.TARGET],  # target
#                                    test_size=0.2,  # percentage of obs in test set
#                                    random_state=0)
#     pipe= joblib.load(filename = config.PIPELINE_NAME)
#
#     # make predictions for test set
#     class_ = pipe.predict(X_test)
#
#     # determine mse and rmse
#     print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
#     print()
#
# if __name__ =="__main__":
#     predict()
