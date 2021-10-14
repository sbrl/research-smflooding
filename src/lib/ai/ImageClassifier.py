import os
import io
import sys
import json

import logging
import tensorflow as tf

from ..polyfills.io import write_file_sync
from ..io.summarywriter import summarywriter, summarylogger
from ..io.settings import settings_get
from .model_lstm import make_model_lstm
from .model_transformer import make_model_transformer

from .AIModel import AIModel
from .LayerPositionEmbedding import LayerPositionEmbedding
from .LayerVisionTransformerEncoder import LayerVisionTransformerEncoder
from .LayerCCTConvEmbedding import LayerCCTConvEmbedding
from .LayerSequencePooling import LayerSequencePooling


class ImageClassifier(AIModel):
	"""Core Compact Convolutional Transformer-based model for classifying images."""
	
	def __init__(self, container, settings, filepath_checkpoint = None):
		"""
		Initialises a new ImageClassifier.
		container(dictionary): The container holding runtime settings and dependencies (i.e. a simple prototype dependency injection system).
		filepath_checkpoint(string?): Optional. If specified, the model in the specified checkpoint file is loaded. If not specified, then a brand-new model will be created with the given settings.
		"""
		super().__init__(container, filepath_checkpoint)
		
		self.settings = settings
	
	
	def custom_layers(self):
		return {
			# Tell Tensorflow about our custom layers so that it can deserialise models that use them
			"LayerPositionEmbedding": LayerPositionEmbedding,
			"LayerVisionTransformerEncoder": LayerVisionTransformerEncoder,
			"LayerCCTConvEmbedding": LayerCCTConvEmbedding,
			"LayerSequencePooling": LayerSequencePooling
		}
	
	
	def make_model(self):
		"""Reinitialises the model."""
		
		model = make_model_cct(**self.settings.model)
		
		logging.info(f"Built cct model")
		
		model.compile(
			optimizer="Adam",
			loss="CategoricalCrossentropy",
			metrics=["accuracy"],
			# Raise this to do multiple batches per execution - good for smaller models
			# Unfortunately this requires specifying number of items in the dataset
			steps_per_execution = 1
		)
		logging.info("Model compiled step 1 / 2")
		# Unsure if this is actually necessary
		# model.build((
		# 	None,
		# 	self.settings.data.sequence_length,
		# 	self.container["glove_word_vector_length"]
		# ))
		logging.info("Model compiled step 2 / 2")
		
		return model
	
	
	def predict_class_ids(self, data, batch_size=None, min_confidence=0.5):
		"""Makes a prediction, but returns the class ids instead of the probabilities."""
		
		predictions = self.predict(data, batch_size)
		result = []
		for item in predictions:
			max_value = -1
			max_index = -1
			for i in range(0, predictions.shape[-1]):
				value = item[i]
				if value > max_value:
					max_index = i
					max_value = value
			
			if max_value > min_confidence:
				result.append(max_index)
			else:
				result.append(None)
		
		return result
