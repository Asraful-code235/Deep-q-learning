import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model


class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network with L2 regularization
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu', kernel_regularizer=regularizers.l2(0.01))(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        outputs = layers.Dense(self._output_dim, activation='linear', kernel_regularizer=regularizers.l2(0.01))(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=keras.losses.mean_squared_error, optimizer=Adagrad(lr=self._learning_rate))
        return model
    

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)


    def train_batch(self, states, q_sa, validation_states, validation_q_sa):
        """
        Train the nn using the updated q-values and stop based on validation loss
        """
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
        self._model.fit(states, q_sa, batch_size=self._batch_size, epochs=50, verbose=0,
                        validation_data=(validation_states, validation_q_sa), callbacks=[early_stop])


    def save_model(self, path):
        """
        Save the current model in the folder as an h5 file and a model architecture summary as a png
        """
        model_file_path = os.path.join(path, 'trained_model.h5')
        self._model.save(model_file_path)
        
        model_structure_path = os.path.join(path, 'model_structure.png')
        plot_model(self._model, to_file=model_structure_path, show_shapes=True, show_layer_names=True)
        
        print("Model saved successfully.")

