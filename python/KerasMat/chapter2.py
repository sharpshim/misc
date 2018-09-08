from keras import models, layers

class ANN_models_class(models.Model):
    def __init__(self, Nin, Nh, Nout):
        # prepare networ layers and activate function
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation("relu")
        softmax = layers.Activation("softmax")

        # connect network elements
        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x, y)
        self.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


import numpy as np
from keras import datasets
from keras.utils import np_utils

def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)

import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc=0)

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc=0)

def main():
    # initialize
    Nin = 784
    Nh = 100
    Nout = 10

    # model
    model = ANN_models_class(Nin, Nh, Nout)
    # data
    (X_train, Y_train), (X_test, Y_test) = Data_func()
    # train
    history = model.fit(X_train, Y_train, batch_size=100, epochs=5, validation_split=0.1)
    # evaluate
    performance_test = model.evaluate(X_test, Y_test, batch_size=100)
    print ("test Loss and Accuracy ->", performance_test)

    # plot
    plot_acc(history)
    plt.show
    plot_loss(history)
    plt.show()

if __name__ == "__main__":
    main()