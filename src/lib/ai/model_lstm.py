import sys
from loguru import logger

import tensorflow as tf


def make_model_lstm(settings, _container):
    """
    Creates an LSTM-based tweet classification AI model.
    settings: The settings object to use to create the model.
    _container: The dynamic container object that contains runtime settings.
    """
    # Useful link: https://github.com/fgafarov/learn-neural-networks/blob/master/sequence_classification_LSTM.py
    model = tf.keras.Sequential()
    i = -1
    layer_count = len(settings.model.lstm_units)
    for units in settings.model.lstm_units:
        i += 1
        lstm = None
        logger.info(f"DEBUG i {i}, layer_count {layer_count}\n")
        if i != layer_count - 1:
            logger.info(f"make_model_lstm: Adding LSTM layer with {units} units")
            lstm = tf.keras.layers.LSTM(units, return_sequences=True)
        else:
            logger.info(f"make_model_lstm: Adding final LSTM layer with {units} units")
            lstm = tf.keras.layers.LSTM(units)
        
        if settings.model.bidirectional:
            logger.info("make_model_lstm: Adding Bidirectional wrapper")
            lstm = tf.keras.layers.Bidirectional(lstm)
        model.add(lstm)
        if settings.model.batch_normalisation:
            logger.info("make_model_lstm: Adding batch normalisation layer")
            model.add(tf.keras.layers.BatchNormalization())
    
    model.add(tf.keras.layers.Dense(settings.data.categories, activation = "softmax"))
    
    return model
