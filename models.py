from keras.models import Sequential
from keras.layers import Dense, LSTM, ConvLSTM2D, BatchNormalization, Dropout, Flatten, Reshape, Bidirectional
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, Callback, ModelCheckpoint

def cnn_blstm():
    
    """ 
    Keras Implementation of CNN-BLSTM, model awarded second place in IFMBE 
    Scientific Challenge 2019 for a P300 classification task.
    
    https://link.springer.com/chapter/10.1007/978-3-030-31635-8_224

    The model has 3 blocks:
        - CNN block to extract spatial features
        - Bidirectional LSTM block to extract temporal features
        - Dense layer with sigmoid activation for classification
    
    The model assumes epochs of 250 samples and 8 EEG channels.
    
    Author: Eduardo Santamaría-Vázquez
    
    """
    
    # Parameters
    n_filt_conv1 = 32
    ker_size_conv1 = 4
    stride_conv1 = 4
    drop_conv1 = 0.1
    
    n_neurons_lstm1 = 16
    drop_lstm1 = 0.1
    recurrent_drop_lstm1 = 0.0
    
    n_neurons_lstm2 = 8
    drop_lstm2 = 0.1
    recurrent_drop_lstm2 = 0.0
    
    # Keras model
    model = Sequential()
    # Convolutional Layer
    model.add(BatchNormalization())
    model.add(Conv1D(filters=n_filt_conv1,
                      kernel_size=ker_size_conv1, 
                      strides=stride_conv1,
                      activation='relu',
                      kernel_initializer='he_normal',
                      input_shape=(250,8,1)))
    model.add(Dropout(drop_conv1))
    # LSTM layer
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(n_neurons_lstm1,
                                 return_sequences=True,
                                 dropout=drop_lstm1,
                                 recurrent_dropout=recurrent_drop_lstm1)))
    # LSTM layer
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(n_neurons_lstm2,
                                 return_sequences=False,
                                 dropout=drop_lstm2,
                                 recurrent_dropout=recurrent_drop_lstm2)))
    #Block3
    model.add(Dense(1, activation='sigmoid'))
    return model


def cnn_lstm():
    
    """ 
    Keras Implementation of CNN-LSTM.
    
    https://link.springer.com/chapter/10.1007/978-3-030-31635-8_224

    The model has 3 blocks:
        - CNN block to extract spatial features
        - LSTM block to extract temporal features
        - Dense layer with sigmoid activation for classification
    
    The model assumes epochs of 250 samples and 8 EEG channels.
    
    Author: Eduardo Santamaría-Vázquez
    
    """
    
    # Parameters
    n_filt_conv1 = 32
    ker_size_conv1 = 4
    stride_conv1 = 4
    drop_conv1 = 0.1
    
    n_neurons_lstm1 = 16
    drop_lstm1 = 0.1
    recurrent_drop_lstm1 = 0.0
    
    n_neurons_lstm2 = 8
    drop_lstm2 = 0.1
    recurrent_drop_lstm2 = 0.0
    
    # Keras model
    model = Sequential()
    # Convolutional Layer
    model.add(BatchNormalization())
    model.add(Conv1D(filters=n_filt_conv1,
                      kernel_size=ker_size_conv1, 
                      strides=stride_conv1,
                      activation='relu',
                      kernel_initializer='he_normal',
                      input_shape=(250,8,1)))
    model.add(Dropout(drop_conv1))
    # LSTM layer
    model.add(BatchNormalization())
    model.add(LSTM(n_neurons_lstm1,
                   return_sequences=True,
                   dropout=drop_lstm1,
                   recurrent_dropout=recurrent_drop_lstm1))
    # LSTM layer
    model.add(BatchNormalization())
    model.add(LSTM(n_neurons_lstm2,
                   return_sequences=False,
                   dropout=drop_lstm2,
                   recurrent_dropout=recurrent_drop_lstm2))
    #Block3
    model.add(Dense(1, activation='sigmoid'))
    return model




