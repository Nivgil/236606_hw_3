import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

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


def transfer_weights(source_model, replace_fc=True, suffix=''):
    model = keras.models.Sequential()

    for layer in source_model.model.layers[:-2]:
        layer.trainable = False
        # layer.name = '%s_%s' % (layer.name, suffix)
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

    return x_train, y_train, x_test, y_test


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


def embedding_tests(data, source_model):
    X_train, y_train, X_test, y_test = data
    y_train = np.argmax(y_train, 1)
    y_test = np.argmax(y_test, 1)

    source_model.model.layers.pop()
    source_model.model.layers.pop()
    model = source_model.model
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    X_test_features = model.predict(X_test)
    X_train_features = model.predict(X_train)
    scores = {}

    for train_size in (100, 1000, 10000):
        X_train_small, _, y_train_small, _ = train_test_split(X_train_features, y_train,
                                                              train_size=train_size, random_state=42, stratify=y_train)
        knn_classifier = KNeighborsClassifier(n_neighbors=6)
        knn_classifier.fit(X_train_small, y_train_small)
        accuracy = knn_classifier.score(X_test_features, y_test)
        print(
            'training set size {} samples,\t\tknn classification accuracy - [{:.3f}%]'.format(len(X_train_small),
                                                                                              accuracy * 100))
        scores[train_size] = accuracy

    return scores


def main():
    x_train, y_train, x_test, y_test = load_data()

    source_model = cifar100vgg(train=False)
    x_train = source_model.normalize_production(x_train)
    x_test = source_model.normalize_production(x_test)

    embedding_tests((x_train, y_train, x_test, y_test), source_model)


if __name__ == '__main__':
    main()
