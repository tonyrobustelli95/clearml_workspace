from clearml import Task
import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    # Initialize task
    task = Task.init(project_name='clearml-init', task_name='Esecuzione 2')
    tf.config.set_visible_devices([], 'GPU')

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = np.array(x_train / 255.0), np.array(x_test / 255.0)

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),tf.keras.layers.Dense(128, activation='relu'), tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'])
             
    model.fit(x_train, y_train, epochs=10)
