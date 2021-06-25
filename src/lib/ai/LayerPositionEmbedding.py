import tensorflow as tf


class LayerPositionEmbedding(tf.keras.layers.Layer):
    """
    Embedding layer that adds positional information to the input tensor.
    This is important for Transformers, as they otherwise have no way to obtain
    such positional information.
    """
    
    def __init__(self, max_length, embed_dim_count):
        """Initialises a new LayerPositionEmbedding layer instance."""
        self(LayerPositionEmbedding, self).__init__()
        self.embedding_positions = tf.keras.layers.Embedding(
            input_dim = max_length,
            output_dim = embed_dim_count
        )
        
    
    def call(self, tensor_in):
        """Runs the specified tensor through the model."""
        input_shape = tf.shape(tensor_in)
        positions = self.embedding_positions(tf.range(
            start=0,
            limit=input_shape[-1],
            delta=1
        ))
        
        return tensor_in + positions
