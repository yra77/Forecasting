
import os
import IPython
import IPython.display
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.python.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential

#%matplotlib inline

sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 14, 8

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


df = pd.read_csv('BTC-USD.csv', parse_dates=['Date'])

df.head()

ax = df.plot(x='Date', y='Close');
ax.set_xlabel("Date")
ax.set_ylabel("Close Price (USD)")
plt.show()

scaler = MinMaxScaler()
close_price = df.Close.values.reshape(-1, 1)
scaled_close = scaler.fit_transform(close_price)

scaled_close.shape

np.isnan(scaled_close).any()

scaled_close = scaled_close[~np.isnan(scaled_close)]

scaled_close = scaled_close.reshape(-1, 1)

np.isnan(scaled_close).any()



SEQ_LEN = 100

def to_sequences(data, seq_len):
    d = []

    for index in range(len(data) - seq_len):
        d.append(data[index: index + seq_len])

    return np.array(d)

def preprocess(data_raw, seq_len, train_split):

    data = to_sequences(data_raw, seq_len)

    num_train = int(train_split * data.shape[0])

    X_train = data[:num_train, :-1, :]
    y_train = data[:num_train, -1, :]

    X_test = data[num_train:, :-1, :]
    y_test = data[num_train:, -1, :]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = preprocess(scaled_close, SEQ_LEN, train_split = 0.95)


print(X_train.shape)

print(X_test.shape)


DROPOUT = 0.2
WINDOW_SIZE = SEQ_LEN - 1

model = keras.Sequential()

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=True),
                        input_shape=(WINDOW_SIZE, X_train.shape[-1])))#,activation='tanh',recurrent_activation='sigmoid'
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM((WINDOW_SIZE * 2), return_sequences=True)))
model.add(Dropout(rate=DROPOUT))

model.add(Bidirectional(LSTM(WINDOW_SIZE, return_sequences=False)))

model.add(Dense(units=1))

model.add(Activation('linear'))


model.compile(
    loss='mean_squared_error', 
    optimizer='adam'
)

BATCH_SIZE = 64

history = model.fit(
    X_train, 
    y_train, 
    epochs=15, 
    batch_size=BATCH_SIZE, 
    shuffle=False,
    validation_split=0.1
)

model.evaluate(X_test, y_test)

#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

y_hat = model.predict(X_test)

model.save('Model')
model.save("Model/model_26_07.h5")

y_test_inverse = scaler.inverse_transform(y_test)
y_hat_inverse = scaler.inverse_transform(y_hat)
 
plt.plot(y_test_inverse, label="Actual Price", color='green')
plt.plot(y_hat_inverse, label="Predicted Price", color='red')
 
plt.title('Bitcoin price prediction')
plt.xlabel('Time [days]')
plt.ylabel('Price')
plt.legend(loc='best')
 
plt.show();




#Variant tensorflow

#import os
#import datetime

#import IPython
#import IPython.display
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd
#import seaborn as sns
#import tensorflow as tf

#mpl.rcParams['figure.figsize'] = (8, 6)
#mpl.rcParams['axes.grid'] = False


## Import data
#df = pd.read_csv('btc1.csv')

## slice [start:stop:step], starting from index 5 take every 6th record.
##df = df[5::6]
#date_time = df.pop('DATE')#, format='%d.%m.%Y %H:%M:%S')
#timestamp_s = date_time
##print(timestamp_s)

#df.head()


#plot_cols = ['BTC']
#plot_features = df[plot_cols]
#plot_features.index = date_time
#_ = plot_features.plot(subplots=True)

#plot_features = df[plot_cols][:480]
#plot_features.index = date_time[:480]
#_ = plot_features.plot(subplots=True)

##plt.show()

#df.describe().transpose()


#day = 24*60*60
#year = (365.2425)*day

#df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
#df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
#df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
#df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))


#column_indices = {name: i for i, name in enumerate(df.columns)}

#n = len(df)
#train_df = df[0:int(n*0.7)]
#val_df = df[int(n*0.7):int(n*0.9)]
#test_df = df[int(n*0.9):]

#num_features = df.shape[1]



#train_mean = train_df.mean()
#train_std = train_df.std()

#train_df = (train_df - train_mean) / train_std
#val_df = (val_df - train_mean) / train_std
#test_df = (test_df - train_mean) / train_std




#class WindowGenerator():
#  def __init__(self, input_width, label_width, shift,
#               train_df=train_df, val_df=val_df, test_df=test_df,
#               label_columns=None):
#    # Store the raw data.
#    self.train_df = train_df
#    self.val_df = val_df
#    self.test_df = test_df

#    # Work out the label column indices.
#    self.label_columns = label_columns
#    if label_columns is not None:
#      self.label_columns_indices = {name: i for i, name in
#                                    enumerate(label_columns)}
#    self.column_indices = {name: i for i, name in
#                           enumerate(train_df.columns)}

#    # Work out the window parameters.
#    self.input_width = input_width
#    self.label_width = label_width
#    self.shift = shift

#    self.total_window_size = input_width + shift

#    self.input_slice = slice(0, input_width)
#    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

#    self.label_start = self.total_window_size - self.label_width
#    self.labels_slice = slice(self.label_start, None)
#    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

#  def __repr__(self):
#    return '\n'.join([
#        f'Total window size: {self.total_window_size}',
#        f'Input indices: {self.input_indices}',
#        f'Label indices: {self.label_indices}',
#        f'Label column name(s): {self.label_columns}'])



#def split_window(self, features):
#  inputs = features[:, self.input_slice, :]
#  labels = features[:, self.labels_slice, :]
#  if self.label_columns is not None:
#    labels = tf.stack(
#        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
#        axis=-1)

#  # Slicing doesn't preserve static shape information, so set the shapes
#  # manually. This way the `tf.data.Datasets` are easier to inspect.
#  inputs.set_shape([None, self.input_width, None])
#  labels.set_shape([None, self.label_width, None])

#  return inputs, labels

#WindowGenerator.split_window = split_window



#w1 = WindowGenerator(input_width=720, label_width=720, shift=24,
#                     label_columns=['BTC'])
#w1
#w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
#                     label_columns=['BTC'])
#w2


## Stack three slices, the length of the total window:
#example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
#                           np.array(train_df[100:100+w2.total_window_size]),
#                           np.array(train_df[200:200+w2.total_window_size])])


#example_inputs, example_labels = w2.split_window(example_window)

#print('All shapes are: (batch, time, features)')
#print(f'Window shape: {example_window.shape}')
#print(f'Inputs shape: {example_inputs.shape}')
#print(f'labels shape: {example_labels.shape}')

#w2.example = example_inputs, example_labels

#def plot(self, model=None, plot_col='BTC', max_subplots=3):
#  inputs, labels = self.example
#  plt.figure(figsize=(12, 8))
#  plot_col_index = self.column_indices[plot_col]
#  max_n = min(max_subplots, len(inputs))
#  for n in range(max_n):
#    plt.subplot(max_n, 1, n+1)
#    plt.ylabel(f'{plot_col} [normed]')
#    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
#             label='Inputs', marker='.', zorder=-10)

#    if self.label_columns:
#      label_col_index = self.label_columns_indices.get(plot_col, None)
#    else:
#      label_col_index = plot_col_index

#    if label_col_index is None:
#      continue

#    plt.scatter(self.label_indices, labels[n, :, label_col_index],
#                edgecolors='k', label='Labels', c='#2ca02c', s=64)
#    if model is not None:
#      predictions = model(inputs)
#      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
#                  marker='X', edgecolors='k', label='Predictions',
#                  c='#ff7f0e', s=64)

#    if n == 0:
#      plt.legend()

#  plt.xlabel('Time [h]')
 
#WindowGenerator.plot = plot


#w2.plot()
#w2.plot(plot_col='BTC')


#def make_dataset(self, data):
#  data = np.array(data, dtype=np.float32)
#  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
#      data=data,
#      targets=None,
#      sequence_length=self.total_window_size,
#      sequence_stride=1,
#      shuffle=True,
#      batch_size=32,)

#  ds = ds.map(self.split_window)

#  return ds

#WindowGenerator.make_dataset = make_dataset


#@property
#def train(self):
#  return self.make_dataset(self.train_df)

#@property
#def val(self):
#  return self.make_dataset(self.val_df)

#@property
#def test(self):
#  return self.make_dataset(self.test_df)

#@property
#def example(self):
#  """Get and cache an example batch of `inputs, labels` for plotting."""
#  result = getattr(self, '_example', None)
#  if result is None:
#    # No example batch was found, so get one from the `.train` dataset
#    result = next(iter(self.train))
#    # And cache it for next time
#    self._example = result
#  return result

#WindowGenerator.train = train
#WindowGenerator.val = val
#WindowGenerator.test = test
#WindowGenerator.example = example


## Each element is an (inputs, label) pair
#w2.train.element_spec

#for example_inputs, example_labels in w2.train.take(1):
#  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#  print(f'Labels shape (batch, time, features): {example_labels.shape}')


#  single_step_window = WindowGenerator(
#    input_width=1, label_width=1, shift=1,
#    label_columns=['BTC'])
#single_step_window


#for example_inputs, example_labels in single_step_window.train.take(1):
#  print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
#  print(f'Labels shape (batch, time, features): {example_labels.shape}')



#class Baseline(tf.keras.Model):

#     def __init__(self, label_index=None):
#          super().__init__()
#          self.label_index = label_index
#     def call(self, inputs):
#        if self.label_index is None:
#         return inputs
#        result = inputs[:, :, self.label_index]
#        return result[:, :, tf.newaxis]

#baseline = Baseline(label_index=column_indices['BTC'])

#baseline.compile(loss=tf.losses.MeanSquaredError(),
#                 metrics=[tf.metrics.MeanAbsoluteError()])

#val_performance = {}
#performance = {}
#val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
#performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

#wide_window = WindowGenerator(
#    input_width=24, label_width=24, shift=1,
#    label_columns=['BTC'])

#wide_window
#print('Input shape:', wide_window.example[0].shape)
#print('Output shape:', baseline(wide_window.example[0]).shape)
#wide_window.plot(baseline)
#plt.show()

# END  Variant tensorflow


# Variant №1 no working

## Drop date variable
#data = data.drop(['DATE'], 1)

## Dimensions of dataset
#n = data.shape[0]
#p = data.shape[1]

## Make data a np.array
#data = data.values

##plt.plot(data)
##plt.show()

## Training and test data
#train_start = 0
#train_end = int(np.floor(0.8*n))
#test_start = train_end - 500
#test_end = n
#data_train = data[np.arange(train_start, train_end), :]
#data_test = data[np.arange(test_start, test_end), :]

## Scale data
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaler.fit(data_train)
#data_train = scaler.transform(data_train)
#data_test = scaler.transform(data_test)

## Build X and y
#X_train = data_train[:, 1:]
#y_train = data_train[:, 0]
#X_test = data_test[:, 1:]
#y_test = data_test[:, 0]

## Number of stocks in training data
#n_stocks = X_train.shape[1]

## Neurons
#n_neurons_1 = 1024
#n_neurons_2 = 512
#n_neurons_3 = 256
#n_neurons_4 = 128

## Session
#net = tf.compat.v1.Session()


## Placeholder
#X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_stocks])
#Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])

## Initializers
#sigma = 1
#weight_initializer = tf.compat.v1.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
#bias_initializer = tf.compat.v1.zeros_initializer()

## Hidden weights
#W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
#bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
#W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
#bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
#W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
#bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
#W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
#bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

## Output weights
#W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
#bias_out = tf.Variable(bias_initializer([1]))

## Hidden layer
#hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
#hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
#hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
#hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

## Output layer (transpose!)
#out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

## Cost function
#mse = tf.reduce_mean(tf.compat.v1.squared_difference(out, Y))

## Optimizer
#opt = tf.compat.v1.train.AdamOptimizer().minimize(mse)

## Init
#net.run(tf.compat.v1.global_variables_initializer())

## Setup plot
#plt.ion()
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#line1, = ax1.plot(y_test)
#line2, = ax1.plot(y_test * 0.5)
#plt.show()

## Fit neural net
#batch_size = 256
#mse_train = []
#mse_test = []

## Run
#epochs = 100
#for e in range(epochs):

#    # Shuffle training data
#  #  shuffle_indices = np.random.permutation(np.arange(len(y_train)))
#   # X_train = X_train[shuffle_indices]
#    #y_train = y_train[shuffle_indices]

#    # Minibatch training
#    for i in range(0, len(y_train)):
#        start = i * batch_size
#        batch_x = X_train[start:start + batch_size]
#        batch_y = y_train[start:start + batch_size]
#        # Run optimizer with batch
#        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

#        # Show progress
#        if np.mod(i, 50) == 0:
#            # MSE train and test
#            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
#            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
#            print('MSE Train: ', mse_train[-1])
#            print('MSE Test: ', mse_test[-1])
#            # Prediction
#            pred = net.run(out, feed_dict={X: X_test, Y: y_test})
#            line2.set_ydata(pred)
#            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
#            plt.pause(0.01)

#         #Print final MSE after Training

#mse_final = net.run(mse, feed_dict= {X: X_test, Y: y_test})
#print(mse_final)