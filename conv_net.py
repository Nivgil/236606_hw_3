import keras
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, save
from bokeh.models import NumeralTickFormatter
from datetime import datetime
import os
import densenet

OUTPUT_DIR = 'ConvModels'
BATCH_SIZE = 64
EPOCHS = 300


def lr_scheduler(epoch):
    if epoch < 150:
        return 0.1
    elif epoch < 225:
        return 0.1 * 0.1
    else:
        return 0.1 * 0.1 * 0.1


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


def channelwise_normalization(x, mean, std):
    assert len(x.shape) == 4

    x = x.astype(np.float32)

    for i in range(x.shape[3]):
        x[:, :, :, i] -= mean[i]
        x[:, :, :, i] /= std[i]

    return x


def load_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1)

    mean = np.zeros(X_train.shape[-1])
    std = np.zeros(X_train.shape[-1])

    # for i in range(X_train.shape[-1]):
    #     mean[i] = np.mean(X_train[:, :, :, i])
    #     std[i] = np.std(X_train[:, :, :, i])

    mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
    std = [x / 255.0 for x in [63.0, 62.1, 66.7]]

    X_train, X_test = channelwise_normalization(X_train, mean, std), \
                      channelwise_normalization(X_test, mean, std)

    return X_train, y_train, X_test, y_test

    # Normalize by channel instead
    # X_train, X_valid, X_test = channelwise_normalization(X_train, mean, std), \
    #                            channelwise_normalization(X_valid, mean, std), \
    #                            channelwise_normalization(X_test, mean, std)

    # return X_train, y_train, X_valid, y_valid, X_test, y_test


def build_model():
    model = densenet.DenseNet(classes=10, input_shape=(32, 32, 3), depth=19, growth_rate=12,
                              bottleneck=True, reduction=0.5)

    model.compile(optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=False),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    return model


def output_params(model, dir_path):
    with open('%s/params.txt' % dir_path, 'w') as f:
        model.summary(print_fn=lambda s: f.write("%s\n" % s))
        f.write("epochs: %d" % EPOCHS)


def main():
    np.random.seed(42)
    tf.set_random_seed(42)

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = "%s/%s" % (OUTPUT_DIR, date)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    X_train, y_train, X_test, y_test = load_data()
    model = build_model()

    output_params(model, dir_path)

    datagen = ImageDataGenerator(rotation_range=0,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)

    datagen.fit(X_train)

    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=42),
                                  epochs=EPOCHS,
                                  steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                  validation_data=(X_test, y_test),
                                  callbacks=[keras.callbacks.LearningRateScheduler(lr_scheduler),
                                             keras.callbacks.ModelCheckpoint('%s/model.h5' % dir_path,
                                                                             save_best_only=True,
                                                                             save_weights_only=False,
                                                                             monitor='val_acc')])

    test_res = model.evaluate(X_test, y_test)
    with open("%s/test_results.txt" % dir_path, 'w') as f:
        f.write("%s" % test_res)
    print(test_res)

    with open("%s/history.txt" % dir_path, 'w') as f:
        f.write("%s" % history.history)
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    train_loss = history.history['loss']
    train_acc = history.history['acc']
    p_accuracy = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                        x_axis_label='epochs', y_axis_label='Error Rate',
                        title="Training & Test Accuracy", x_axis_type='linear')
    p_loss = figure(plot_width=600, plot_height=600, min_border=10, min_border_left=50,
                    x_axis_label='epochs', y_axis_label='Error Rate',
                    title="Training & Test Loss", y_axis_type='log')
    p_accuracy.background_fill_color = "#fafafa"
    p_loss.background_fill_color = "#fafafa"
    p_accuracy.yaxis[0].formatter = NumeralTickFormatter(format="0.0%")
    p_accuracy.line(range(1, len(train_loss) + 1), val_acc, line_width=3, line_dash='solid',
                    legend='Validation Accuracy',
                    line_color='blue')
    p_accuracy.line(range(1, len(train_loss) + 1), train_acc, line_width=3, line_dash='solid', legend='Train Accuracy',
                    line_color='orange')
    p_loss.line(range(1, len(train_loss) + 1), val_loss, line_width=3, line_dash='solid', legend='Validation Loss',
                line_color='blue')
    p_loss.line(range(1, len(train_loss) + 1), train_loss, line_width=3, line_dash='solid', legend='Train Loss',
                line_color='orange')
    # plot_history(history, dir_path)
    p_accuracy.legend.location = 'bottom_right'
    output_file('accuracy.html')
    save(p_accuracy)
    output_file('loss.html')
    save(p_loss)


if __name__ == '__main__':
    main()
