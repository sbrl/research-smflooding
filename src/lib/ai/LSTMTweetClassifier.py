import os

import logging
import tensorflow as tf

from ..io.settings import settings_get


class LSTMTweetClassifier:
    """Core LSTM-based model to classify tweets."""
    
    def __init__(self):
        """Initialises a new LSTMTweetClassifier."""
        self.settings = settings_get()
        self.make_model()
        
        self.dir_tensorboard = os.path.join(self.settings.output, "tensorboard")
        self.dir_checkpoints = os.path.join(self.settings.output, "checkpoints")
        self.filepath_tsvlog = os.path.join(self.settings.output, "metrics.tsv")
        
        
        if not os.path.exists(self.dir_checkpoints):
            os.makedirs(self.dir_checkpoints, 0o750)
    
    
    def make_model(self):
        """Reinitialises the model."""
        # TODO: Implement a model here.
        # Useful link: https://github.com/fgafarov/learn-neural-networks/blob/master/sequence_classification_LSTM.py
        self.model = tf.keras.Sequential()
        for units in self.settings.model.lstm_units:
            logging.info(f"LSTMTweetClassifier: Adding LSTM layer with {units}")
            self.model.add(tf.keras.layers.LSTM(units))
        self.model.add(tf.keras.layers.Dense(self.settings.data.categories, activation = "softmax"))
        self.model.compile(
            optimizer="Adam",
            loss="CategoricalCrossentropy",
            steps_per_execution = 1 # Raise this to do multiple batches per execution - good for smaller models
        )
    
    def make_callbacks(self):
        """Generates a list of callbacks to be called when a model is training."""
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(
                    self.dir_checkpoints,
                    "checkpoint_e{epoch:d}_acc{val_acc:.3f}.hdf5"
                ),
                monitor="val_acc"
            ),
            tf.keras.callbacks.CSVLogger(
                filename=self.filepath_tsvlog,
                separator="\t"
            ),
            tf.keras.callbacks.ProgBarLogger(),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.dir_tensorboard,
                histogram_freq=1,
                write_images=True,
                update_freq="epoch"
            )
        ]
    
    def train(self, data):
        """Trains the model on the given data."""
        return self.model.fit(
            data,
            # The batch size is specified as part of the keras.utils.Sequence/dataset/generator objetc
            epochs = self.settings.train.epochs,
            validation_split=self.settings.train.validation_data_percent,
            callbacks=self.make_callbacks()
        )
    
    def evaluate(self, data):
        """Evaluates the given input data with the AI model, but does not update any weights."""
        return self.model.evaluate(
            data
        )
