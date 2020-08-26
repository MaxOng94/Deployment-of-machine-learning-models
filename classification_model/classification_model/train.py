import config
from pipeline import pipe
import preprocessors as pf
import pandas as pd

from sklearn.model_selection import train_test_split

import joblib

def run_training():
    """
    To run training that is the same as our jupyter notebook

    """
    data = pd.read_csv(config.TRAINING_DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(config.TARGET, axis=1),  # predictors
                                    data[config.TARGET],  # target
                                    test_size=0.2,  # percentage of obs in test set
                                    random_state=0)
    pipe.fit(X_train,y_train)
    joblib.dump(pipe, config.PIPELINE_NAME)
    print("training done!")



if __name__ == "__main__":
    run_training()
