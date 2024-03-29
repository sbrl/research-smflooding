import tensorflow as tf


class LayerSequencePooling(tf.keras.layers.Layer):
	
	def __init__(self, activation="softmax", **kwargs):
		super(LayerSequencePooling, self).__init__(**kwargs)
		
		self.activation=activation
		
		self.layernorm_a = tf.keras.layers.LayerNormalization(epsilon=1e-5)
		self.dense = tf.keras.layers.Dense(1)
	
	
	def get_config(self):
		"""
		Returns this layer's configuration so that it can be serialised by Tensorflow.
		Without this, Tensorflow will crash on saving a checkpoint O.o
		"""
		return {
			"activation": self.activation,
		}
	
	
	def call(self, inputs):
		
		out_layernorm = self.layernorm_a(inputs)
		
		out_attention = self.dense(inputs)
		
		out_softmax = tf.nn.softmax(out_attention, axis=1)
		
		out_weighted = tf.matmul(
			out_attention,
			out_layernorm,
			transpose_a=True
		)
		
		return tf.squeeze(out_weighted, -2)
