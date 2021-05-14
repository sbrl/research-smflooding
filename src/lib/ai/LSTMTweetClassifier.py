import tensorflow as tf


class LSTMTweetClassifier:
    """Core LSTM-based model to classify tweets."""
    
    def __init__(self, sequence_length = 100):
        """Initialises a new LSTMTweetClassifier."""
        self.sequence_length = sequence_length
        self.reset()
    
    def reset():
        """Reinitialises the model."""
        # TODO: Implement a model here.
        # Useful link: https://github.com/fgafarov/learn-neural-networks/blob/master/sequence_classification_LSTM.py
