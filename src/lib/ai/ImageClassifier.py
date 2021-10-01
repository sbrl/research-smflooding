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
from .LayerTransformerBlock import LayerTransformerBlock


class ImageClassifier(AIModel):
	"""Core Compact Convolutional Transformer-based model for classifying images."""
	
	def __init__(self, container, filepath_checkpoint = None):
		"""Initialises a new ImageClassifier."""
		super().__init__(self, container, filepath_checkpoint)
	
    
	def custom_layers(self):
        super().custom_layers()
        return {
			# Tell Tensorflow about our custom layers so that it can deserialise models that use them
			"LayerPositionEmbedding": LayerPositionEmbedding,
			"LayerTransformerBlock": LayerTransformerBlock
		}
	
    
	def make_model(self):
		"""Reinitialises the model."""
        
        model = make_model_cct(self.settings, self.container)
		
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
		model.build((
			None,
			self.settings.data.sequence_length,
			self.container["glove_word_vector_length"]
		))
		logging.info("Model compiled step 2 / 2")
		
        return model
	
	
	def train(self, data_train, data_validate):
		"""Trains the model on the given data."""
		return self.model.fit(
			data_train,
			validation_data=data_validate,
			# The batch size is specified as part of the keras.utils.Sequence/dataset/generator object
			epochs = self.settings.train.epochs,
			callbacks=self.make_callbacks()
		)
	
	def predict(self, data, batch_size=None):
		"""Makes a prediction for the given input data with the AI model, but does not update any weights."""
		return self.model.predict(
			data,
			batch_size=batch_size
		)
	
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
