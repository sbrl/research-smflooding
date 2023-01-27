import tensorflow as tf


class LayerTransformerBlock(tf.keras.layers.Layer):
	"""Implements a single encoder block from the Transformer model architecture."""
	
	def __init__(self, units_embedding, attention_heads_count, units_dense, dropout = 0.1, **kwargs):
		"""
		Implements a Transformer encoder as per the Attention Is All You Need paper.
		units_embedding: The number of units to use in the embedding dimension.
		units_dense: The number of units in the internal dense layer.
		attention_heads_count: The number of attention heads to create.
		dropout: The dropout percentage, from 0 to 1.
		"""
		super(LayerTransformerBlock, self).__init__(**kwargs)
		
		self.units_embedding = units_embedding
		self.attention_heads_count = attention_heads_count
		self.units_dense = units_dense
		self.dropout = dropout
		
		self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
			num_heads = attention_heads_count,
			key_dim=units_embedding
		)
		self.dense = tf.keras.Sequential([
			tf.keras.layers.Dense(units_dense, activation="gelu"),
			tf.keras.layers.Dense(units_embedding)
		])
		self.layer_normalisation_a = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.layer_normalisation_b = tf.keras.layers.LayerNormalization(epsilon=1e-6)
		self.dropout_a = tf.keras.layers.Dropout(dropout)
		self.dropout_b = tf.keras.layers.Dropout(dropout)
	
	def get_config(self):
		"""
		Returns this layer's configuration so that it can be serialised by Tensorflow.
		Without this, Tensorflow will crash on saving a checkpoint O.o
		"""
		return {
			"units_embedding": self.units_embedding,
			"attention_heads_count": self.attention_heads_count,
			"units_dense": self.units_dense,
			"dropout": self.dropout
		}
	
	def call(self, inputs, training):
		"""Runs the given inputs through the model."""
		out_attention = self.multi_head_attention(inputs, inputs)
		out_attention = self.dropout_a(out_attention, training=training)
		out_attention = self.layer_normalisation_a(inputs + out_attention)
		out_end = self.dense(out_attention)
		out_end = self.dropout_b(out_end, training=training)
		return self.layer_normalisation_b(out_attention + out_end)
