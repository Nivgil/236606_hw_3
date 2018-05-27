import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

from cifar100vgg import cifar100vgg

CLASSES = 10
EPOCHS = 5


def plot_history(history, dir_path, baseline=None):
    his = history.history
    val_acc = his['val_acc']
    train_acc = his['acc']
    plt.plot(np.arange(len(val_acc)), val_acc, label='val_acc')
    plt.plot(np.arange(len(train_acc)), train_acc, label='acc')
    if baseline is not None:
        his = baseline.history
        val_acc = his['val_acc']
        train_acc = his['acc']
        plt.plot(np.arange(len(val_acc)), val_acc, label='baseline val_acc')
        plt.plot(np.arange(len(train_acc)), train_acc, label='baseline acc')
    plt.legend()
    plt.savefig('%s/plot.png' % dir_path)

    plt.show()


def transfer_weights(source_model, replace_fc=True):
    model = keras.models.Sequential()
    for layer in source_model.model.layers[:-2]:
        layer.trainable = False
        model.add(layer)

    if replace_fc is True:
        model.add(keras.layers.Dense(CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])

    return model


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1)

    # return x_train, y_train, x_valid, y_valid, x_test, y_test
    return x_train, y_train, x_test, y_test, x_test, y_test


def fine_tuning(data, source_model, batch_size):
    X_train, y_train, X_test, y_test = data

    model = transfer_weights(source_model)

    hist = model.fit(X_train,
                     y_train,
                     epochs=EPOCHS,
                     validation_data=(X_test, y_test),
                     batch_size=batch_size,
                     )

    return hist


def fine_tuning_tests(data, source_model):
    X_train, y_train, X_test, y_test = data

    histories = {}

    for train_size, batch_size in [(100, 1), (1000, 32), (10000, 64)]:
        X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train,
                                                              train_size=train_size, random_state=42, stratify=y_train)

        histories[train_size] = fine_tuning((X_train_small, y_train_small, X_test, y_test), source_model, batch_size)

    return


def embedding(data, source_model):
    X_train, y_train, X_test, y_test = data

    batch_size = 128
    model = transfer_weights(source_model, replace_fc=False)
    X_train_features = model._predict(X_train)
    import ipdb; ipdb.set_trace()
    # hist = model.fit(X_train,
    #                  y_train,
    #                  epochs=EPOCHS,
    #                  validation_data=(X_test, y_test),
    #                  batch_size=batch_size,
    #                  )


def main():
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data()

    source_model = cifar100vgg(train=False)
    # model = transfer_weights(source_model)
    x_train, x_valid, x_test = source_model.normalize_production(x_train), \
                               source_model.normalize_production(x_valid), \
                               source_model.normalize_production(x_test)

    # fine_tuning_tests((x_train, y_train, x_test, y_test), model)
    embedding((x_train, y_train, x_test, y_test), source_model)


if __name__ == '__main__':
    main()
