import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import os
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import keras.backend as K

from cifar100vgg import cifar100vgg

CLASSES = 10
EPOCHS = 10
BASE_LR = 0.1
WEIGHTS_LAMBDA = 1.0
WEIGHT_DECAY = 0.0005
RATE = 0.95


def lr_scheduler(epoch):
    return BASE_LR * (RATE ** epoch)


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


class Transfer_Regularization(keras.regularizers.Regularizer):
    """Regularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, weights):
        self.weights = K.cast_to_floatx(weights)

    def __call__(self, x):
        regularization = K.sum(WEIGHTS_LAMBDA * K.square(self.weights - x))
        return regularization

    def get_config(self):
        return {'weights_lamb': WEIGHTS_LAMBDA}


def transfer_weights(source_model, replace_fc=True, transfer_regularization=False):
    model = keras.models.Sequential()

    for layer in source_model.model.layers[:-2]:
        layer.trainable = transfer_regularization
        model.add(layer)

    if transfer_regularization:
        for layer in model.layers:
            if layer.__class__.__name__ in ["Conv2D", "Dense"]:
                layer.add_loss(Transfer_Regularization(layer.get_weights()[0])(layer.weights[0]))
                layer.add_loss(Transfer_Regularization(layer.get_weights()[1])(layer.weights[1]))

    if replace_fc is True:
        model.add(keras.layers.Dense(CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=BASE_LR, momentum=0.9, nesterov=True, clipnorm=0.5),
                  metrics=['accuracy'])

    return model


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def fine_tuning(data, source_model, batch_size, is_regularized=False, epochs=EPOCHS, validation=False):

    X_train, y_train, X_test, y_test = data

    model = transfer_weights(source_model, transfer_regularization=is_regularized)

    hist = model.fit(X_train,
                     y_train,
                     epochs=epochs,
                     validation_data=(X_test[:500], y_test[:500]) if validation else None,
                     batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler)],
                     verbose=2,
                     )

    print("Fine tuning for %d examples: %f" % (X_train.shape[0], model.evaluate(X_test, y_test, verbose=0)[1]))

    del model

    return hist


def fine_tuning_tests(data, source_model, is_regularized=False, epochs=EPOCHS, validation=False):

    print("######################")
    print("# Fine Tuning Tests: #")
    print("######################")
    print()

    X_train, y_train, X_test, y_test = data

    histories = {}

    for train_size, batch_size in [(100, 10), (1000, 32), (10000, 64)]:
        X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train,
                                                              train_size=train_size, random_state=42, stratify=y_train)

        dir_path = "Transfer_learning/%d" % train_size
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        histories[train_size] = fine_tuning((X_train_small, y_train_small, X_test, y_test),
                                            source_model,
                                            batch_size,
                                            is_regularized,
                                            epochs,
                                            validation)
        # if validation:
        #     plot_history(histories[train_size], dir_path)

    return


def embedding_logistic_regression(data, source_model):

    print("########################################")
    print("# Embedding Logistic Regression Tests: #")
    print("########################################")
    print()

    X_train, y_train, X_test, y_test = data

    #batch_size = 128
    source_model.model.layers.pop()
    source_model.model.layers.pop()
    source_model.model.layers.pop()
    source_model.model.layers.pop()
    source_model.model.layers.pop()
    source_model.model.layers.pop()

    model = source_model.model
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    X_test_features = model.predict(X_test)

    for train_size in [100, 1000, 10000]:
        print("for %d examples" % train_size)
        X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train,
                                                              train_size=train_size, random_state=42, stratify=y_train)

        X_train_features = model.predict(X_train_small)

        lr = LogisticRegression()
        lr.fit(X_train_features, np.argmax(y_train_small, axis=1))
        print(lr.score(X_test_features, np.argmax(y_test, axis=1)))


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
    np.random.seed(42)
    tf.set_random_seed(42)

    x_train, y_train, x_test, y_test = load_data()

    source_model = cifar100vgg(train=False)
    x_train = source_model.normalize_production(x_train)
    x_test = source_model.normalize_production(x_test)

    fine_tuning_tests((x_train, y_train, x_test, y_test), source_model)
    embedding_logistic_regression((x_train, y_train, x_test, y_test), source_model)
    embedding_tests((x_train, y_train, x_test, y_test), source_model)
    fine_tuning_tests((x_train, y_train, x_test, y_test), source_model, True, 25, True)


if __name__ == '__main__':
    main()
