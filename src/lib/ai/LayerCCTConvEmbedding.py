import tensorflow as tf

class LayerCCTConvEmbedding(tf.keras.layers.Layer):
	
	def __init__(self, filters=64, kernel=7, strides=(1, 1), pool_kernel=3, pool_stride=2, **kwargs):
		"""Initialises a new LayerCCTConvEmbedding layer instance."""
		super(LayerCCTConvEmbedding, self).__init__(**kwargs)
		
		self.filters = filters
		self.kernel = kernel
		self.strides = strides
		self.pool_kernel = pool_kernel
		self.pool_stride = pool_stride
		
		self.submodel = tf.keras.Sequential()
		
		self.submodel.add(tf.keras.layers.Conv2D(
			filters=self.filters, 
			kernel_size=self.kernel,
			strides=self.strides,
			use_bias=False
		))
		self.submodel.add(tf.keras.layers.ReLU())
		# The Keras tutorial adds ZeroPadding2D here, but the original doesn't? I'm confused
		# Ref https://github.com/SHI-Labs/Compact-Transformers/blob/main/src/utils/tokenizer.py
		self.submodel.add(tf.keras.layers.MaxPool2D(
			pool_size=self.pool_kernel,
			strides=self.pool_stride,
			padding="same"
		))
	
	
	
	def get_config(self):
		"""
		Returns this layer's configuration so that it can be serialised by Tensorflow.
		Without this, Tensorflow will crash on saving a checkpoint O.o
		"""
		return {
			"filters": self.filters,
			"kernel": self.kernel,
			"strides": self.strides,
			"pool_kernel": self.pool_kernel,
			"pool_stride": self.pool_stride
		}
	
	def call(self, images):
		return self.submodel(images)
	
	# Positional embeddings are optional apparently
	# The above Keras blog post implements positional embeddings here, but
	# it would really be better suited to a dedicated layer.
