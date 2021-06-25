
import logging

import tensorflow as tf
from LayerPositionEmbedding import LayerPositionEmbedding
from LayerTransformerBlock import LayerTransformerBlock


def make_model_transformer(settings, container):
    """
    Creates a Transformer-based tweet classification model.
    Important: Call this *after* you've set up the data processing pipeline.
    This is becqause the model depends on knowing the number of items in the
    each of the glove embedding elements.
    Ref https://keras.io/examples/nlp/text_classification_with_transformer/
    settings: The settings object to use to create the model.
    container: The dynamic container object that contains runtime settings.
    """
    
    layer_in = tf.keras.layers.Input(shape=(settings.data.sequence_length,))
    layer_next = LayerPositionEmbedding(
        max_length=settings.data.sequence_length,
        embed_dim_count=container["glove_word_vector_length"]
    )(layer_in)
    
    for params in settings.model.transformer_units:
        layer_next = LayerTransformerBlock(
            units_embedding=container["glove_word_vector_length"],
            attention_heads_count=params["attention_heads"],
            units_dense=params["units_dense"],
            dropout=params["dropout"] or 0.1
        )(layer_next)
    
    layer_next = tf.keras.layers.GlobalAveragePooling1D()(layer_next)
    
    layer_next = tf.keras.layers.Dropout(settings.model.dropout)(layer_next)
    layer_next = tf.keras.layers.Dense(
        settings.model.units_last,
        activation="relu"
    )(layer_next)
    
    layer_next = tf.keras.layers.Dropout(settings.model.dropout)(layer_next)
    layer_next = tf.keras.layers.Dense(
        settings.data.categories,
        activation = "softmax"
    )(layer_next)
    
    return tf.keras.Model(
        inputs=layer_in,
        outputs=layer_next
    )
