
import logging

import tensorflow as tf
from .LayerPositionEmbedding import LayerPositionEmbedding
from .LayerVisionTransformerEncoder import LayerVisionTransformerEncoder
from .LayerCCTConvEmbedding import LayerCCTConvEmbedding
from .LayerSequencePooling import LayerSequencePooling

# Default settings = CCT-7/3x1
def make_model_cct(class_count, batch_size=128,
	layers_embed=[ { filters: 64, kernel: 3 } ]
	schotastic_depth_keep
	trans_attention_heads=6,
	layers_transformer = [ { attention_heads: 4, units_dense: 256, copies: 7 } ],
	stochastic_survivability=0.9¸
	image_size=128, image_channels=3, patch_size=16,):
	"""
	Creates a Compact Convolutional Transformer-based image classification model.
	Important: Call this *after* you've set up the data processing pipeline.
	This is because the model depends on knowing the number of items in the
	each of the glove embedding elements.
	Ref https://keras.io/examples/nlp/text_classification_with_transformer/
	image_size (int): The size of the image the model will consume.
	image_channels (int): The number of channels input images will have.
	patch_size (int): The size of the patches to split images up into. The image_size must be a multiple of this number.
	class_count (int): The number of different distinct classes to predict for.
	"""
	
	
	
	layer_in = tf.keras.layers.Input(batch_size=settings.train.batch_size, shape=(
		image_size,
		image_size,
		image_channels
	))
	layer_next = layer_in
	
	if not isinstance(layers_embed, list):
		layers_embed = [ layers_embed ]
	
	
	for layer in layers_embed:
		layer_next = LayerCCTConvEmbedding(**layer)(layer_next)
	
	
	shape = tf.shape(layer_next)
	# Flatten into a sequence
	# Ref https://keras.io/examples/vision/cct/#the-cct-tokenizer
	layer_next = tf.layers.Reshape(
		# ???, sequence length, filters
		(-1, shape[1] * shape[2], tf.shape[-1])
	)(layer_next)
	
	
	# The number of output filters from the Conv2D-based embedding layer(s)
	units_embedding = layers_embed[-1]["filters"]
	
	# OPTIONAL: Position encoding here
	# NOTE: ~~May not be~~ probably isn't identical to the position encoding for the regular transformer
	# layer_next = LayerPositionEmbedding()(layer_in)
	
	stochastic_probabilities = [ x for x in np.linspace(0, stochastic_survivability, len(layers_transformer))]
	
	# THEN: our transformer
	# The paper suggests these should be done in parallel, but the actual implementation doesn't do this
	for i in range(len(layers_transformer)):
		params = layers_transformer[i]
		
		copies = params["copies"]
		params.pop("copies")
		
		for i in range(copies):
			attention_heads_count = params["attention_heads"]
			units_dense = params["units_dense"]
			dropout = params["dropout"] or 0.1 # NOTE: The original paper distinguishes between the MultiHeadAttention dropout & the dense layer dropout
			 = params["stochastic_survivability"] or 0.9 # The probability to KEEP, not drop!
			logging.info(f"make_model_transformer: Adding vision transformer encoding block ("
				+ f"units_embedding = {units_embedding}, "
				+ f"attention heads = {attention_heads_count}, "
				+ f"units_dense = {units_dense}, "
				+ f"dropout = {dropout}, "
				+ f"stochastic_survivability = {stochastic_survivability})")
			layer_next = LayerVisionTransformerEncoder(
				units_embedding=units_embedding
				stochastic_survivability=stochastic_probabilities[i],
				**params
			)(layer_next)
	
	
	
	# NEXT: Sequence pooling
	layer_next = LayerSequencePooling()(layer_next)
	
	
	# FINALLY: Normal linear classifier [layer normalisation → Dense we assume, if pytorch Linear = Tensorflow Dense]
	layer_next = tf.keras.layers.Dense(class_count)(layer_next) # MAY need an activation function, but unclear atm (default: None)
	
	return tf.keras.Model(
		inputs=layer_in,
		outputs=layer_next
	)
