
import logging

import tensorflow as tf
from .LayerPositionEmbedding import LayerPositionEmbedding
from .LayerTransformerBlock import LayerTransformerBlock


def make_model_cct(settings, container,
	trans_attention_heads=6, trans_dropout=0.1,
	image_size=128, image_channels=3, patch_size=16, class_count):
	"""
	Creates a Compact Convolutional Transformer-based image classification model.
	Important: Call this *after* you've set up the data processing pipeline.
	This is because the model depends on knowing the number of items in the
	each of the glove embedding elements.
	Ref https://keras.io/examples/nlp/text_classification_with_transformer/
	settings: The settings object to use to create the model.
	container: The dynamic container object that contains runtime settings.
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
	
	# TODO: Implement the Conv2d-based patch generator here
	# Conv2d [filters=64, activation=ReLU] → Max Pooling → Reshape
	# # Conv2d activation may go separately after Conv2d - not sure - have to look this up as to whether it's different or not
	
	# OPTIONAL: Position encoding here
	# NOTE: ~~May not be~~ probably isn't identical to the position encoding for the regular transformer
	# layer_next = LayerPositionEmbedding()(layer_in)
	
	
	# THEN: our transformer
	# NOTE: Implementation looks slightly different, but we may be able to get awawy with our current implementation
	for params in settings.model.transformer_units:
		units_embedding = container["glove_word_vector_length"]
		attention_heads_count = params["attention_heads"]
		units_dense = params["units_dense"]
		dropout = params["dropout"] or 0.1
		logging.info(f"make_model_transformer: Adding transformer encoding block ("
			+ f"units_embedding = {units_embedding}, "
			+ f"attention heads = {attention_heads_count}, "
			+ f"units_dense = {units_dense}, "
			+ f"dropout = {dropout})")
		layer_next = LayerTransformerBlock(
			units_embedding=units_embedding,
			attention_heads_count=attention_heads_count,
			units_dense=units_dense,
			dropout=dropout
		)(layer_next)
	
	# NEXT: Sequence pooling
	
	# FINALLY: Normal linear classifier [layer normalisation → Dense we assume, if pytorch Linear = Tensorflow Dense]
	
	layer_next = tf.keras.layers.GlobalAveragePooling1D()(layer_next)
	
	layer_next = tf.keras.layers.Dropout(settings.model.dropout)(layer_next)
	layer_next = tf.keras.layers.Dense(
		settings.model.transformer_units_last,
		activation="relu"
	)(layer_next)
	logging.info(f"make_model_transformer: Adding dropout layer (rate = {settings.model.dropout}")
	logging.info(f"make_model_transformer: Adding dense layer (units = {settings.model.transformer_units_last}, activation = relu)")
	
	layer_next = tf.keras.layers.Dropout(settings.model.dropout)(layer_next)
	layer_next = tf.keras.layers.Dense(
		settings.data.categories,
		activation = "softmax"
	)(layer_next)
	logging.info(f"make_model_transformer: Adding dropout layer (rate = {settings.model.dropout})")
	logging.info(f"make_model_transformer: Adding dense layer (units = {settings.data.categories}, activation = softmax)")
	
	return tf.keras.Model(
		inputs=layer_in,
		outputs=layer_next
	)
