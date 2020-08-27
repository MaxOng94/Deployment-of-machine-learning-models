# all the data relevant functions to be placed here
# load data from csv
# save pipeline_to_persist here
# load pipeline function here
import pandas as pd
import joblib

from classification_model.config import config

from sklearn.pipeline import Pipeline

from classification_model.pipeline import pipe


def load_data(*, file_name : str) -> pd.DataFrame:
    data = pd.read_csv(file_name)
    return data

def save_pipeline(*, pipeline_to_persist) -> None:
    save_file_name = "classification_model.pkl"
    save_path = config.TRAIN_MODEL_DIR / save_file_name

    joblib.dump(pipeline_to_persist,save_path)

    print("Pipeline saved")

def load_pipeline(*, file_name : str) -> Pipeline:

    save_path = config.TRAIN_MODEL_DIR / file_name
    pipe = joblib.load(filename =save_path)

    return pipe
