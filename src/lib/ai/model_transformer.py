
from loguru import logger

import tensorflow as tf
from .LayerPositionEmbedding import LayerPositionEmbedding
from .LayerTransformerBlock import LayerTransformerBlock


def make_model_transformer(settings, container):
	"""
	Creates a Transformer-based tweet classification model.
	Important: Call this *after* you've set up the data processing pipeline.
	This is because the model depends on knowing the number of items in the
	each of the glove embedding elements.
	Ref https://keras.io/examples/nlp/text_classification_with_transformer/
	settings: The settings object to use to create the model.
	container: The dynamic container object that contains runtime settings.
	"""
	
	layer_in = tf.keras.layers.Input(batch_size=settings.train.batch_size, shape=(
		settings.data.sequence_length,
		container["glove_word_vector_length"]
	))
	layer_next = LayerPositionEmbedding()(layer_in)
	
	for params in settings.model.transformer_units:
		units_embedding = container["glove_word_vector_length"]
		attention_heads_count = params["attention_heads"]
		units_dense = params["units_dense"]
		dropout = params["dropout"] or 0.1
		logger.info(f"make_model_transformer: Adding transformer encoding block ("
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
	
	layer_next = tf.keras.layers.GlobalAveragePooling1D()(layer_next)
	
	layer_next = tf.keras.layers.Dropout(settings.model.dropout)(layer_next)
	layer_next = tf.keras.layers.Dense(
		settings.model.transformer_units_last,
		activation="gelu"
	)(layer_next)
	logger.info(f"make_model_transformer: Adding dropout layer (rate = {settings.model.dropout}")
	logger.info(f"make_model_transformer: Adding dense layer (units = {settings.model.transformer_units_last}, activation = gelu)")
	
	layer_next = tf.keras.layers.Dropout(settings.model.dropout)(layer_next)
	layer_next = tf.keras.layers.Dense(
		settings.data.categories,
		activation = "softmax"
	)(layer_next)
	logger.info(f"make_model_transformer: Adding dropout layer (rate = {settings.model.dropout})")
	logger.info(f"make_model_transformer: Adding dense layer (units = {settings.data.categories}, activation = softmax)")
	
	return tf.keras.Model(
		inputs=layer_in,
		outputs=layer_next
	)
