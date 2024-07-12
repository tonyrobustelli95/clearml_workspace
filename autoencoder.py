import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Flatten, UpSampling2D, Reshape, Input # type: ignore
import pandas as pd
import numpy as np
from clearml import Task, Dataset, TaskTypes

params = {
    'optimizer': 'adam',
    'loss_function': 'mse',
    'epochs': 5,
    'batch_size': 128,
    'metrics': ['mse','accuracy'],
    'filters': 10
}

def build_model(inputShape):

    encoder = build_encoder(inputShape)
    encoder.summary()
    decoder = build_decoder()
    decoder.summary()

    model = Sequential()
    model.add(encoder)
    model.add(decoder)

    return model

def build_encoder(inputShape):

    enc = Sequential(name="enc_1")

    enc.add(Input((inputShape)))
    enc.add(Conv2D(filters=params['filters'],kernel_size=((2,2)),activation="relu",padding="same"))
    enc.add(Flatten())

    return enc

def build_decoder():

    dec = Sequential(name="dec_1")

    dec.add(Input((3*2*params['filters'],)))
    dec.add(Reshape((3,2,params['filters'])))
    dec.add(Conv2D(filters=1,kernel_size=((2,2)),activation="relu",padding="same"))

    return dec


if __name__ == '__main__':

    task = Task.init(project_name='clearml-init', task_name='Autoencoder training', task_type=TaskTypes.training)
    
    tf.config.set_visible_devices([], 'GPU')

    # load dataset from clearml
    dataset = Dataset.get('25e9281ff0c44754ade05e4d16bf1292',alias='dataset7030')
    file_path = dataset.get_local_copy()

    df = pd.read_csv(file_path + "/" + dataset.list_files()[1])
    del df['category']
    train = np.array(df.to_numpy())
    train = train.reshape(train.shape[0],3,2,1)

    inputShape = (train.shape[1],train.shape[2],train.shape[3])

    # Log parameters
    task.connect(params)

    ae = build_model(inputShape)

    ae.compile(optimizer=params['optimizer'], loss=params['loss_function'], metrics=params['metrics'])

    # Callback ClearML per loggare le metriche
    class ClearMLCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                for metric, value in logs.items():
                    task.get_logger().report_scalar(title=metric, series='training', value=value, iteration=epoch)

    ae.fit(x=train,y=train,validation_data=(train,train),epochs=params['epochs'],
           batch_size=params['batch_size'],callbacks=[ClearMLCallback()])

    # Log model
    base_path = '/Users/antoniorobustelli/Desktop/MLOps/clearml_workspace/models/'
    model_path = 'ae_version' + str(task.id) + '.keras'
    ae.save(base_path + model_path)

    task.upload_artifact('ae_version' + str(task.id), model_path)
    task.get_logger().report_text('Model weights logged successfully.')

    task.close()
