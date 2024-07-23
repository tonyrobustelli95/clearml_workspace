import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Flatten, MaxPool2D, Dense, Reshape, Input # type: ignore
import pandas as pd
import numpy as np
from clearml import Task, Dataset, TaskTypes
import matplotlib.pyplot as plt

params = {
    'optimizer': 'adam',
    'loss_function': 'mse',
    'epochs': 1,
    'batch_size': 128,
    'metrics': ['accuracy'],
    'filters': 10,
    'latent_dim': 4
}

def printHistory(history):

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylim(0, 1)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train loss', 'test loss'], loc='upper right')

    return plt

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
    enc.add(MaxPool2D((2,2)))
    enc.add(Flatten())
    enc.add(Dense(params['latent_dim']))

    return enc

def build_decoder():

    dec = Sequential(name="dec_1")

    dec.add(Input((params['latent_dim'],)))
    dec.add(Dense(3*2*params['filters']))
    dec.add(Reshape((3,2,params['filters'])))
    dec.add(Conv2DTranspose(filters=params['filters'],kernel_size=((2,2)),activation="relu",padding="same"))
    dec.add(Conv2DTranspose(filters=1,kernel_size=((2,2)),activation="relu",padding="same"))

    return dec

if __name__ == '__main__':

    task = Task.init(project_name='clearml-init', task_name='Autoencoder training', task_type=TaskTypes.training)
    task.execute_remotely(queue_name="default")
    
    tf.config.set_visible_devices([], 'GPU')

    # load dataset from clearml
    dataset = Dataset.get('e3c4b1f5295f4210a18fc9fd7a49b089',alias='dataset7030')
    file_path = dataset.get_local_copy()

    df_train = pd.read_csv(file_path + "/" + dataset.list_files()[1])
    del df_train['category']
    train = np.array(df_train.to_numpy())
    train = train.reshape(train.shape[0],3,2,1)

    df_test = pd.read_csv(file_path + "/" + dataset.list_files()[0])
    del df_test['category']
    test = np.array(df_test.to_numpy())
    test = test.reshape(test.shape[0],3,2,1)

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

    history = ae.fit(x=train,y=train,validation_data=(test,test),epochs=params['epochs'],
           batch_size=params['batch_size'],callbacks=[ClearMLCallback()])
    
    task.get_logger().report_matplotlib_figure('Loss Plot', 'Train vs Test loss', printHistory(history).gcf())

    # Log model
    base_path = '/Users/antoniorobustelli/Desktop/MLOps/clearml_workspace/models/'
    model_path = 'ae_version' + str(task.id) + '.keras'
    ae.save(base_path + model_path)

    task.upload_artifact('ae_version' + str(task.id), artifact_object=ae)
    task.get_logger().report_text('Task completed.')

    task.close()
