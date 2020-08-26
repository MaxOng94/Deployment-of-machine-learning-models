from classification_model.config import config
from classification_model.pipeline import pipe
from classification_model.preprocessing import preprocessors as pf
import pandas as pd

from sklearn.model_selection import train_test_split

import joblib

def save_pipeline(*, pipeline_to_persist) -> None:
    save_file_name = "classification_model.pkl"
    save_path = config.TRAIN_MODEL_DIR / save_file_name

    joblib.dump(pipeline_to_persist,save_path)

    print("Pipeline saved")

def run_training() -> None:
    """
    To run training that is the same as our jupyter notebook

    """
    data = pd.read_csv(config.TRAINING_DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(config.TARGET, axis=1),  # predictors
                                    data[config.TARGET],  # target
                                    test_size=0.2,  # percentage of obs in test set
                                    random_state=0)
    pipe.fit(X_train,y_train)
    save_pipeline(pipeline_to_persist = pipe)
    print("training done!")



if __name__ == "__main__":
    run_training()
