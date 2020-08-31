# all the data relevant functions to be placed here
# load data from csv
# save pipeline_to_persist here
# load pipeline function here
import pandas as pd
import joblib

from classification_model.config import config

from sklearn.pipeline import Pipeline

from classification_model.pipeline import pipe

import logging

from classification_model import __version__ as _version

# we want the logger object that we created in the __init__.py of classification_model
_logger= logging.getLogger("classification_model")

def load_data(*, file_name : str) -> pd.DataFrame:
    data = pd.read_csv(file_name)
    return data

def save_pipeline(*, pipeline_to_persist) -> None:
    save_file_name = f"{config.PIPELINE_NAME}{_version}.pkl"
    save_path = config.TRAIN_MODEL_DIR / save_file_name
    remove_old_pipeline(files_to_keep = save_file_name)

    joblib.dump(pipeline_to_persist,save_path)
    _logger.info(f"Pipeline is saved:, {save_file_name}")

    print("Pipeline saved")

def load_pipeline(*, file_name : str) -> Pipeline:

    save_path = config.TRAIN_MODEL_DIR / file_name
    pipe = joblib.load(filename =save_path)

    return pipe

def remove_old_pipeline(*,files_to_keep):
    """
    Remove old model pipelines.
    This is to ensure a simple one-to-one mapping
    between package version n model version to be imported
    n to be used by other applications.

    """
    # config.TRAIN_MODEL_DIR is a directory
    for model_file in config.TRAIN_MODEL_DIR.iterdir():
        # within the TRAIN_MODEL_DIR, if the model is not in these file specified
        if model_file not in [files_to_keep, "__init__.py"]:
            model_file.unlink()
