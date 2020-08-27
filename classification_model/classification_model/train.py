from classification_model.config import config
from classification_model.pipeline import pipe

from classification_model.preprocessing import preprocessors as pf
from classification_model.preprocessing.data_management import save_pipeline
from classification_model.preprocessing.data_management import load_data


from sklearn.model_selection import train_test_split

import joblib
import pandas as pd



def run_training() -> None:
    """
    To run training that is the same as our jupyter notebook

    """
    data = load_data(file_name = config.TRAINING_DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(config.TARGET, axis=1),  # predictors
                                    data[config.TARGET],  # target
                                    test_size=0.2,  # percentage of obs in test set
                                    random_state=0)

    pipe.fit(X_train,y_train)
    save_pipeline(pipeline_to_persist = pipe)
    print("training done!")



if __name__ == "__main__":
    run_training()
