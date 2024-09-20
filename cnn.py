import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Input # type: ignore
import pandas as pd
import numpy as np
from clearml import Task, Dataset, TaskTypes
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from focal_loss import SparseCategoricalFocalLoss

params = {
    'optimizer': 'adam',
    'loss_function': SparseCategoricalFocalLoss(gamma=2.0),
    'epochs': 15,
    'batch_size': 1024,
    'metrics': ['accuracy'],
    'filters': [32,16,8],
    'num_neurons': [256,256,7]
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

    model = Sequential(name="CNN")

    model.add(Input((inputShape)))
    model.add(Conv2D(filters=params['filters'][0],kernel_size=((2,2)),activation="relu",padding="same"))
    model.add(Conv2D(filters=params['filters'][1],kernel_size=((2,2)),activation="relu",padding="same"))
    model.add(Conv2D(filters=params['filters'][2],kernel_size=((2,2)),activation="relu",padding="same"))
    #model.add(MaxPool2D((2,2)))
    model.add(Flatten())

    model.add(Dense(params['num_neurons'][0],activation="sigmoid"))
    model.add(Dense(params['num_neurons'][1],activation="sigmoid"))
    model.add(Dense(params['num_neurons'][2],activation="softmax"))

    return model

if __name__ == '__main__':

    task = Task.init(project_name='clearml-init', task_name='CNN training', task_type=TaskTypes.training)
    #task.execute_remotely(queue_name="default")
    
    tf.config.set_visible_devices([], 'GPU')

    # load dataset from clearml
    dataset = Dataset.get('e3c4b1f5295f4210a18fc9fd7a49b089',alias='dataset7030')
    file_path = dataset.get_local_copy()

    df_train = pd.read_csv(file_path + "/" + dataset.list_files()[1])
    y_train = df_train['category']
    y_train_hot = tf.keras.utils.to_categorical(y_train, num_classes=7)
    del df_train['category']
    train = np.array(df_train.to_numpy())
    scaler = MinMaxScaler()
    train = scaler.fit_transform(train)
    train = train.reshape(train.shape[0],3,2,1)

    df_test = pd.read_csv(file_path + "/" + dataset.list_files()[0])
    y_test = df_test['category']
    y_test_hot = tf.keras.utils.to_categorical(y_test, num_classes=7)
    del df_test['category']
    test = np.array(df_test.to_numpy())
    test = scaler.transform(test)
    test = test.reshape(test.shape[0],3,2,1)

    inputShape = (train.shape[1],train.shape[2],train.shape[3])

    # Log parameters
    task.connect(params)

    cnn = build_model(inputShape)

    cnn.compile(optimizer=params['optimizer'], loss=params['loss_function'], metrics=params['metrics'])

    cnn.summary()

    # Callback ClearML per loggare le metriche
    class ClearMLCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is not None:
                for metric, value in logs.items():
                    task.get_logger().report_scalar(title=metric, series='training', value=value, iteration=epoch)

    history = cnn.fit(x=train,y=y_train,validation_data=(test,y_test),epochs=params['epochs'],
           batch_size=params['batch_size'],callbacks=[ClearMLCallback()])
    
    task.get_logger().report_matplotlib_figure('Loss Plot', 'Train vs Test loss', printHistory(history).gcf())
    
    pred = cnn.predict(test)

    # Print Confusion Matrix
    cm = confusion_matrix(y_test, np.argmax(pred,axis=-1))
    print("Confusion Matrix:")
    print(cm)
    
    # Print Classification Report
    cr = classification_report(y_test, np.argmax(pred,axis=-1))
    print("Classification Report:")
    print(cr)

    # Log model
    base_path = '/Users/antoniorobustelli/Desktop/MLOps/clearml_workspace/models/'
    model_path = 'cnn_version' + str(task.id) + '.keras'
    cnn.save(base_path + model_path)

    task.upload_artifact('cnn_version' + str(task.id), artifact_object=cnn)
    task.get_logger().report_text('Task completed.')

    task.close()

