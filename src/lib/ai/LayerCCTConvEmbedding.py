import tensorflow as tf

class LayerCCTConvEmbedding(tf.keras.layers.Layer):
	
	def __init__(self, filters=64, kernel=7, stride=1, pool_kernel=3, pool_stride=2, **kwargs):
		"""Initialises a new LayerCCTConvEmbedding layer instance."""
		super(LayerCCTConvEmbedding, self).__init__(**kwargs)
		
		self.layers = layers
		
		self.submodel = tf.keras.Sequential()
		
		self.submodel.add(tf.keras.layers.Conv2D(
			filters=filters, 
			kernel_size=kernel,
			stride=stride,
			use_bias=False
		))
		self.submodel.add(tf.keras.layers.ReLU())
		# The Keras tutorial adds ZeroPadding2D here, but the original doesn't? I'm confused
		# Ref https://github.com/SHI-Labs/Compact-Transformers/blob/main/src/utils/tokenizer.py
		self.submodel.add(tf.keras.layers.MaxPool2D(
			pool_size=pool_kernel,
			strides=pool_stride,
			padding="same"
		))
		
	
	def call(self, images):
		return self.submodel(images)
	
	# Positional embeddings are optional apparently
	# The above Keras blog post implements positional embeddings here, but
	# it would really be better suited to a dedicated layer.
