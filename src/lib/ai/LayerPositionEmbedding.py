import tensorflow as tf


class LayerPositionEmbedding(tf.keras.layers.Layer):
    """
    Embedding layer that adds positional information to the input tensor.
    This is important for Transformers, as they otherwise have no way to obtain
    such positional information.
    """
    
    def __init__(self, **kwargs):
        """Initialises a new LayerPositionEmbedding layer instance."""
        super(LayerPositionEmbedding, self).__init__(**kwargs)
        
    
    def get_time_signal_single(self, token_position, embedding_depth_size):
        """
        Calculates a time signal for a given token position with a given embedding depth.
        Note for given input parameters, the output of this function is
        constant.
        token_position: The index of the token in the input sequence. For
            example, if the input sequence is a series of words, the
            token_position would be the position in the sentence (starting from
            zero).
        embedding_depth_size: The size of the embedding dimension. This is
            normally the last dimension in the input tensor's shape. For
            example,positions if I have an input tensor representing the words in a
            sentence of shape [ 32, 10, 50 ], then 32 is the batch size, 10 is
            the token position (i.e. 1 element in that dimension per word in
            the input sentence), and 50 is the size of he embedding dimension,
            which means each word in in input sentence is represented by 50
            different values in the tensor.
        """
        indices = tf.range(tf.constant(0, "float32"), embedding_depth_size) # range = 0, 1, 2, 3, 4, .... embedding_depth_size
        result = tf.math.divide(
            tf.math.multiply(indices, tf.constant(2, "float32")),
            embedding_depth_size
        )
        result = tf.math.pow(tf.constant(10000, dtype="float32"), result)
        result = tf.math.divide(token_position, result)
        
        result = tf.where(indices % 2 == 0,
            tf.sin(result), # Do sin() for even values
            tf.cos(result)  # Do cos() for odd values
        )
        return result
    
    @tf.function
    def get_time_signal(self, number_of_tokens, embedding_depth_size):
        """
        Returns a time signal as a tensor.
        For given set of input arguments, the output of this function is
        constant.
        number_of_tokens: The number of tokens in the input sequence. Also
            known as the sequence length. For example, if the input was a
            series of embeddings representing the words in a sentence, then the
            number of tokens would be the number of words in the sentence (as
            the model sees it, since one has to pad shorter sentences to ensure
            that all sentences are the same length for the model).
        embedding_depth_size: The size fo he embedding dimension. This is
            normally the last dimension in the input tensor's shape. Given an
            input that consists of word-level embeddings for a sentence, then
            the embedding depth size in the size of the dimension that encodes
            information about each individual word. Given an input shape of
            [ 32, 10, 50 ], then 32 would be the batch size, 10 the number of
            tokens or sequence length, and 50 would be the embedding depth
            size.
        """
        values = []
        for token_position in range(0, number_of_tokens):
            values.append(self.get_time_signal_single(token_position, embedding_depth_size))
        
        return tf.stack(values)
    
    
    def call(self, tensor_in):
        """Adds the positional time signal to the input tensor."""
        # print("DEBUG input_shape", tensor_in.shape)
        if len(tensor_in.shape) != 3:
            raise Exception(f"LayerPositionEmbedding: Error: The input tensor has a shape of rank {len(tensor_in.shape)} (specifically {tensor_in.shape}), but a tensor of rank 3 was expected (specifically [ batch_size, sequence_length, embedding_size ])")
        
        batch_size = tensor_in.shape[0]
        if batch_size is None:
            raise Exception("LayerPositionEmbedding: Error: The batch size of the input tensor is None. This is incompatible with this layer, because it makes it impossible to encode the time signal (as far as I know). Please specify the batch size explcitly.")
        number_of_tokens = tensor_in.shape[-2]
        embedding_size = tensor_in.shape[-1]
        positions_single = self.get_time_signal(number_of_tokens, embedding_size)
        # print("DEBUG positions_single shape", positions_single.shape)
        # print("DEBUG batch_size", batch_size)
        positions = tf.stack([ positions_single for i in range(0, batch_size) ])
        # print("DEBUG tensor_in ", tensor_in.shape)
        # print("DEBUG positions ", positions.shape)
        return tensor_in + positions
