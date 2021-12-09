import os
import io
import sys
import json

from loguru import logger
import tensorflow as tf

from ..io.settings import settings_get
from .model_cct import make_model_cct

from .AIModel import AIModel
from .LayerPositionEmbedding import LayerPositionEmbedding
from .LayerVisionTransformerEncoder import LayerVisionTransformerEncoder
from .LayerCCTConvEmbedding import LayerCCTConvEmbedding
from .LayerSequencePooling import LayerSequencePooling
from ..data.load_image import load_image


class ImageClassifier(AIModel):
	"""Core Compact Convolutional Transformer-based model for classifying images."""
	
	def __init__(self, container, class_count, settings, filepath_checkpoint = None):
		"""
		Initialises a new ImageClassifier.
		container(dictionary): The container holding runtime settings and dependencies (i.e. a simple prototype dependency injection system).
		class_count(int): The number of output classes of thing the model should predict.
		filepath_checkpoint(string?): Optional. If specified, the model in the specified checkpoint file is loaded. If not specified, then a brand-new model will be created with the given settings.
		"""
		super().__init__(container, filepath_checkpoint)
		
		self.class_count = class_count
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
		
		if self.settings.model.type == "cct":
			logger.info("Making CCT")
			model = make_model_cct(
				class_count=self.class_count,
				**vars(self.settings.model)
			)
		elif self.settings.model.type == "resnet":
			logger.info("Making ResNet50")
			image_size = self.settings.model.image_size
			if image_size < 32:
				image_size = 32
			
			logger.debug(f"image_size {image_size}, raw {self.settings.model.image_size}")
			model = tf.keras.applications.resnet50.ResNet50(
				classes=self.class_count,
				weights=None, # Could also be "imagenet"
				input_shape=( image_size, image_size, 3 )
			)
		else:
			raise Exception(f"Error: Invalid model type {self.settings.model.type}")
		
		sys.stdout.write(f"Built {self.settings.model.type} model")
		
		model.compile(
			optimizer="Adam",
			loss="CategoricalCrossentropy",
			metrics=["accuracy"],
			# Raise this to do multiple batches per execution - good for smaller models
			# Unfortunately this requires specifying number of items in the dataset
			steps_per_execution = 1
		)
		logger.info("Model compiled step 1 / 2")
		# Unsure if this is actually necessary
		# model.build((
		# 	None,
		# 	self.settings.data.sequence_length,
		# 	self.container["glove_word_vector_length"]
		# ))
		logger.info("Model compiled step 2 / 2")
		
		return model
	
	def predict_class_ids(self, filepaths, batch_size=None, buffer_size=64, min_confidence=0.5):
		"""Makes a prediction, but returns the class ids instead of the probabilities."""
		
		image_size = self.model.input.shape[1]
		
		items_processed = 0
		result = []
		acc = []
		for filepath in filepaths:
			acc.append(filepath)
			
			if len(acc) >= buffer_size:
				data = list(map(lambda filepath_inner : load_image(
					filepath_inner,
					image_size	# Use the model's input shape, as it may be different from the current config file
				), acc))
				
				# Remove any None values where images failed to load
				i = 0
				while i < len(data):
					if data[i] is None:
						data.pop(i)
						acc.pop(i)
					else:
						i = i + 1
				
				# Make predictions
				predictions = self.predict(tf.stack(data), batch_size)
				
				# Process the predictions, add them to the main results array
				for itemi,item in enumerate(predictions):
					max_value = -1
					max_index = -1
					for i in range(0, predictions.shape[-1]):
						value = item[i]
						if value > max_value:
							max_index = i
							max_value = value
					
					if max_value > min_confidence:
						result.append([ acc[itemi], max_index ])
					else:
						result.append(None)
				
				# Progress indicator
				items_processed = items_processed + len(acc)
				sys.stderr.write(f"{items_processed} images processed\r")
				# Clear the accumulator
				acc = []
		
		return result
