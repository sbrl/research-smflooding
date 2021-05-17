import tensorflow as tf


class LSTMTweetClassifier:
    """Core LSTM-based model to classify tweets."""
    
    def __init__(self, settings):
        """Initialises a new LSTMTweetClassifier."""
        self.settings = settings
        self.make_model()
    
    def make_model(self):
        """Reinitialises the model."""
        # TODO: Implement a model here.
        # Useful link: https://github.com/fgafarov/learn-neural-networks/blob/master/sequence_classification_LSTM.py
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(128))
        self.model.add(tf.keras.layers.Dense(self.settings["categories"], activation = "softmax"))
        self.model.compile(
            optimizer="Adam",
            loss="CategoricalCrossentropy",
            steps_per_execution = 1 # Raise this to do multiple batches per execution - good for smaller models
        )
    
    def train(data, settings):
        """Trains the model on the given data."""
        self.model.fit
