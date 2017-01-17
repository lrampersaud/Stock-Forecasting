import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential, model_from_json
from keras.callbacks import Callback
import matplotlib.animation as animation


np.random.seed(1234)

class TrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.predictions = []
        self.i = 0
        self.save_every = 50

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.i += 1
        #if self.i % self.save_every == 0:
            #pred = self.model.predict(X_train)
            #self.predictions.append(pred)


history = TrainingHistory()


def data_power_consumption(path_to_dataset,
                           sequence_length=60,
                           ratio=1.0):
    max_values = ratio * 33852

    with open(path_to_dataset) as f:
        data = csv.reader(f, delimiter=",")
        power = []
        nb_of_values = 0
        for line in data:
            try:
                power.append(float(line[7]))
                nb_of_values += 1
            except ValueError:
                pass
            # 2049280.0 is the total number of valid values, i.e. ratio = 1.0
            if nb_of_values >= max_values:
                break

    print "Data loaded from csv. Formatting..."

    result = []
    for index in range(len(power) - sequence_length):
        result.append(power[index: index + sequence_length])
    result = np.array(result)  # shape (2049230, 50)

    result_mean = result.mean()
    result -= result_mean
    print "Shift : ", result_mean
    print "Data  : ", result.shape

    row = round(0.9 * result.shape[0])
    train = result[:row, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[row:, :-1]
    y_test = result[row:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print "Train on  : ", X_train.shape
    print "Predict on  : ", X_test.shape

    return [X_train, y_train, X_test, y_test]


def build_model():
    model = Sequential()
    layers = [1, 60, 100, 1]

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print "Compilation Time : ", time.time() - start
    return model


def load_model():
    try:
        model = model_from_json(open('my_model_architecture.json').read())
        model.load_weights('my_model_weights.h5')
        model.compile(loss="mse", optimizer="rmsprop")
        return model
    except Exception as e:
        print str(e)
        return None


def save_model(model):
    json_string = model.to_json()
    open('my_model_architecture.json', 'w').write(json_string)
    model.save_weights('my_model_weights.h5')


def run_network(model=None, data=None):
    global_start_time = time.time()
    epochs = 10
    ratio = 0.5
    batch_size = 32
    sequence_length = 60
    path_to_dataset = 'dataApple.csv'

    if data is None:
        print 'Loading data... '
        X_train, y_train, X_test, y_test = data_power_consumption(path_to_dataset, sequence_length, ratio)
    else:
        X_train, y_train, X_test, y_test = data

    print '\nData Loaded. Compiling...\n'
    try:
        if model is None:
            print '\nBuilding Model...\n'
            model = build_model()
            model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs, validation_split=0.05, callbacks=[history])
            print '\nSaving Model...\n'
            # save_model(model)
        else:
            print '\nModel Loaded...\n'
        print '\nPredicting Values...\n'
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print 'Training duration (s) : ', time.time() - global_start_time
        return model, y_test, 0

    try:
        plt.figure(1)
        plt.plot(history.losses, 'b')
        plt.ylabel('error')
        plt.xlabel('iteration')
        plt.title('training error')

        plt.figure(2)
        plt.plot(y_test, 'g', label='real price')
        plt.plot(predicted, 'r', label='predicted price')
        plt.ylabel('price')
        plt.xlabel('time')
        plt.title('predicted stock price')

        plt.figure(3)
        plt.plot(y_train, 'b')
        plt.ylabel('price')
        plt.xlabel('time')
        plt.title('previous stock price')

        plt.show()

    except Exception as e:
        print str(e)
    print 'Training duration (s) : ', time.time() - global_start_time

    return model, y_test, predicted
    #pass


run_network(model=load_model(), data=None)

"""
fig = plt.figure(figsize=(5, 2.5))
plt.plot(X_test, y_test,  label='data')
line, = plt.plot(X_test, history.predictions[0],  label='prediction')
plt.legend(loc='upper left')

def update_line(num):
    plt.title('iteration: {0}'.format((history.save_every * (num + 1))))
    line.set_xdata(X_test)
    line.set_ydata(history.predictions[num])
    return []

ani = animation.FuncAnimation(fig, update_line, len(history.predictions),
                                   interval=50, blit=True)
ani.save('img/neuron.mp4', fps=30, extra_args=['-vcodec', 'libx264', '-pix_fmt','yuv420p'])
plt.close()

plt.figure(figsize=(5, 2.5))
plt.plot(X_test, y_test, label='data')
plt.plot(X_test, history.predictions[0], label='prediction')
plt.legend(loc='upper left')
plt.title('iteration: 0')
plt.savefig('img/neuron_start.png')
plt.close()
"""