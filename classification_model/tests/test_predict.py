from classification_model.predict import predict
from classification_model.config import config
import numpy as np

from classification_model.preprocessing.data_management import load_data, load_pipeline


from sklearn.model_selection import train_test_split
def test_predict():
    # given
    data = load_data(file_name = config.TRAINING_DATA_FILE)
    X_train, X_test, y_train, y_test = train_test_split(data.drop(config.TARGET, axis=1),  # predictors
                                        data[config.TARGET],  # target
                                        test_size=0.2,  # percentage of obs in test set
                                        random_state=0)
    single_test = X_test



    # when
    subject = predict(input_data =single_test )

    # then
    assert subject is not None
    assert isinstance(subject[0], np.int64)
    assert subject[0] == 0
