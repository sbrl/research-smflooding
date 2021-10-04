import tensorflow as tf
import tensorflow_addons as tfa

class LayerVisionTransformerEncoder(tf.keras.layers.Layer):
	"""Implements a single encoder block from the Transformer model architecture."""
	
	def __init__(self, units_embedding, attention_heads_count, units_dense, dropout = 0.1, stochastic_survivability=1, **kwargs):
		"""
		Implements a Transformer encoder as per the Attention Is All You Need paper.
		units_embedding: The number of units to use in the embedding dimension.
		units_dense: The number of units in the internal dense layer.
		attention_heads_count: The number of attention heads to create.
		dropout: The dropout percentage, from 0 to 1.
		"""
		super(LayerVisionTransformerEncoder, self).__init__()
		
		self.units_embedding = units_embedding
		self.attention_heads_count = attention_heads_count
		self.units_dense = units_dense
		self.dropout = dropout
		self.stochastic_survivability = stochastic_survivability
		
		self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
			num_heads=attention_heads_count,
			key_dim=units_embedding,
			dropout=dropout # Regular transformers don't do this (we think)
		)
		
		self.layer_normalisation_a = tf.keras.layers.LayerNormalization(epsilon=1e-5)
		self.layer_normalisation_b = tf.keras.layers.LayerNormalization(epsilon=1e-5)
		
		self.stochastic_a = tfa.layers.StochasticDepth(stochastic_survivability)
		self.stochastic_b = tfa.layers.StochasticDepth(stochastic_survivability)
		
		self.dense = tf.keras.Sequential([
			tf.keras.layers.Dense(units_dense, activation="gelu"), # Gaussian error linear unit
			tf.keras.layers.Dropout(dropout) # Alternative: Alpha Dropout, which keeps the mean & variance the same (though we haven't seen evidence of that being used anywhere)
		])
	
	def get_config(self):
		"""
		Returns this layer's configuration so that it can be serialised by Tensorflow.
		Without this, Tensorflow will crash on saving a checkpoint O.o
		"""
		return {
			"units_embedding": self.units_embedding,
			"attention_heads_count": self.attention_heads_count,
			"units_dense": self.units_dense,
			"dropout": self.dropout,
			"stochastic_survivability": self.stochastic_survivability
		}
	
	def call(self, inputs, training):
		"""Runs the given inputs through the model."""
		
		out_layernorm_a = self.layer_normalisation_a(inputs)
		out_attention = self.multi_head_attention(out_layernorm_a, out_layernorm_a)
		
		out_skipconn_a = self.stochastic_a([out_attention, inputs], training=training)
		
		out_layernorm_b = self.layer_normalisation_b(out_skipconn_a)
		
		out_dense = self.dense(out_layernorm_b)
		
		out_skipconn_b = self.stochastic_b([out_dense, out_skipconn_a], training=training)
		
		return out_skipconn_b
