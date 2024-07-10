import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Flatten, Reshape # type: ignore

def build_model(inputShape):

    encoder = build_encoder(inputShape)
    decoder = build_decoder()

    model = Sequential()
    model.add(encoder)
    model.add(decoder)

    return model

def build_encoder(inputShape):

    enc = Sequential(name="enc_1")

    enc.add(Conv2D(filters=10,kernel_size=((2,2)),activation="relu",input_shape=inputShape))
    enc.add(Flatten())

    return enc

def build_decoder():

    dec = Sequential(name="dec_1")

    dec.add(Reshape((2,2)))
    dec.add(Conv2D(filters=10,kernel_size=((2,2)),activation="relu"))
    dec.add(Conv2D(filters=1,kernel_size=((2,2)),activation="relu"))

    return dec

if __name__ == '__main__':

    inputShape = (10,10,3)

    ae = build_model(inputShape)
    ae.summary()

    ae.compile(optimizer='adam', loss='mse')

