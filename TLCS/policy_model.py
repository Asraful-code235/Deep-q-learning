import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


class PolicyModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._model = self._build_model(num_layers, width)


    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network for policy model
        """
        inputs = keras.Input(shape=(self._input_dim,))
        x = layers.Dense(width, activation='relu')(inputs)
        for _ in range(num_layers):
            x = layers.Dense(width, activation='relu')(x)
        outputs = layers.Dense(self._output_dim, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='policy_model')
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self._learning_rate))
        return model
    

    def predict(self, state):
        """
        Predict the probabilities of each action from a single state
        """
        state = np.reshape(state, [1, self._input_dim])
        return self._model.predict(state)


    def train_batch(self, states, actions):
        """
        Train the nn using the updated actions
        """
        self._model.fit(states, actions, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'policy_model.h5'))


    @property
    def input_dim(self):
        return self._input_dim


    @property
    def output_dim(self):
        return self._output_dim


    @property
    def batch_size(self):
        return self._batch_size
