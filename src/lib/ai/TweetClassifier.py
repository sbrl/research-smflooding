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


class TweetClassifier:
	"""Core LSTM-based model to classify tweets."""
	
	def __init__(self, container, filepath_checkpoint = None):
		"""Initialises a new TweetClassifier."""
		self.container = container
		
		if filepath_checkpoint is None:
			self.settings = settings_get()
			
			self.dir_tensorboard = os.path.join(self.settings.output, "tensorboard")
			self.dir_checkpoints = os.path.join(self.settings.output, "checkpoints")
			self.filepath_tsvlog = os.path.join(self.settings.output, "metrics.tsv")
			self.filepath_summary = os.path.join(self.settings.output, "summary.txt")
			self.filepath_settings = os.path.join(self.settings.output, "settings.toml")
			
			if not os.path.exists(self.dir_checkpoints):
				os.makedirs(self.dir_checkpoints, 0o750)
			
			if not self.container["glove_word_vector_length"]:
				sys.stderr.write("Error: Please initialise the dataset object before initialising the model.\n")
				exit(1)
			
			self.make_model()
		else:
			self.load_model(filepath_checkpoint)
	
	
	def load_model(self, filepath_checkpoint):
		"""
		Loads a saved model from the given filename.
		filepath_checkpoint (string): The filepath to load the saved model from.
		"""
		
		if not os.path.exists(filepath_checkpoint):
			print(f"TweetClassifier Error: No such file or directory {filepath_checkpoint}")
			sys.exit(2)
		
		
		self.model = tf.keras.models.load_model(filepath_checkpoint)
		
	
	def make_model(self):
		"""Reinitialises the model."""
		
		if self.settings.model.type == "lstm":
			self.model = make_model_lstm(self.settings, self.container)
		elif self.settings.model.type == "transformer":
			self.model = make_model_transformer(self.settings, self.container)
		else:
			logging.error(f"Error: Unknown model type '{self.settings.model.type}'")
			sys.exit(1)
		
		logging.info(f"Built model of type '{self.settings.model.type}'")
		
		self.model.compile(
			optimizer="Adam",
			loss="CategoricalCrossentropy",
			metrics=["accuracy"],
			# Raise this to do multiple batches per execution - good for smaller models
			# Unfortunately this requires specifying number of items in the dataset
			steps_per_execution = 1
		)
		logging.info("Model compiled step 1 / 2")
		self.model.build((
			None,
			self.settings.data.sequence_length,
			self.container["glove_word_vector_length"]
		))
		logging.info("Model compiled step 2 / 2")
		
		# Write the settings and the model summary to disk
		write_file_sync(self.filepath_settings, self.settings.source)
		summarywriter(self.model, self.filepath_summary)
		summarylogger(self.model)
		
		logging.info("Model summary above")
	
	
	def make_callbacks(self):
		"""Generates a list of callbacks to be called when a model is training."""
		return [
			tf.keras.callbacks.ModelCheckpoint(
				filepath=os.path.join(
					self.dir_checkpoints,
					"checkpoint_e{epoch:d}_acc{val_accuracy:.3f}.hdf5"
				),
				monitor="val_accuracy"
			),
			tf.keras.callbacks.CSVLogger(
				filename=self.filepath_tsvlog,
				separator="\t"
			),
			tf.keras.callbacks.ProgbarLogger(),
			tf.keras.callbacks.TensorBoard(
				log_dir=self.dir_tensorboard,
				histogram_freq=1,
				write_images=True,
				update_freq=self.settings.train.tensorboard_update_freq
			)
		]
	
	def train(self, data_train, data_validate):
		"""Trains the model on the given data."""
		return self.model.fit(
			data_train,
			validation_data=data_validate,
			# The batch size is specified as part of the keras.utils.Sequence/dataset/generator object
			epochs = self.settings.train.epochs,
			callbacks=self.make_callbacks()
		)
	
	def evaluate(self, data):
		"""Evaluates the given input data with the AI model, but does not update any weights."""
		return self.model.evaluate(
			data
		)
