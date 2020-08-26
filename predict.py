import config
from pipeline import pipe
import preprocessors as pf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import joblib

import pandas as pd

def predict():
    data = pd.read_csv(config.TRAINING_DATA_FILE)
    X_train,X_test,y_train,y_test = train_test_split(data.drop(config.TARGET, axis=1),  # predictors
                                   data[config.TARGET],  # target
                                   test_size=0.2,  # percentage of obs in test set
                                   random_state=0)
    pipe= joblib.load(filename = config.PIPELINE_NAME)

    # make predictions for test set
    class_ = pipe.predict(X_test)

    # determine mse and rmse
    print('test accuracy: {}'.format(accuracy_score(y_test, class_)))
    print()

if __name__ =="__main__":
    predict()
