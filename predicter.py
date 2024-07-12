from clearml import Task, Model, Dataset
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

if __name__ == '__main__':

    # load dataset from clearml
    dataset = Dataset.get('25e9281ff0c44754ade05e4d16bf1292',alias='dataset7030')
    file_path = dataset.get_local_copy()

    df = pd.read_csv(file_path + "/" + dataset.list_files()[0])
    del df['category']
    test = np.array(df.to_numpy())
    test = test.reshape(test.shape[0],3,2,1)

    # Retrieve the model by using the model ID (no task_id)
    model_id = '7b340e98d8d44424afff8a3b8f1ab630'
    model = Model(model_id=model_id)

    # Download the model to a local path
    model_path = model.get_local_copy()
    print(f'Model is downloaded to: {model_path}')

    ae = load_model(model_path)

    encoder = ae.get_layer('enc_1')

    encoder.summary()
    
    """
    Predictions related to latent space

    prediction = encoder.predict(test)
    print(prediction)
    """

    