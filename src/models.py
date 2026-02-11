import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D

def build_cnn_model(hp, maxlen, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1,  # +1 for padding
                        output_dim=hp.Choice('embedding_dim', [8, 16, 32]),
                        input_length=maxlen))

    model.add(Conv1D(filters=hp.Int('filters', 32, 128, step=32),
                     kernel_size=hp.Choice('kernel_size', [3, 5, 7]),
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1)))
    model.add(GlobalMaxPooling1D())

    model.add(Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def build_lstm_model(hp, maxlen, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size + 1,
                        output_dim=hp.Choice('embedding_dim', [8, 16, 32]),
                        input_length=maxlen))
    
    # Original LSTM notebook logic
    model.add(Bidirectional(LSTM(units=hp.Int('lstm_units', 32, 128, step=32), return_sequences=False)))
    model.add(Dropout(hp.Float('dropout_rate', 0.2, 0.5, step=0.1)))
    
    model.add(Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
