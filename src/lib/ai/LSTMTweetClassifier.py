import os
import io
import sys
import json

import logging
import tensorflow as tf

from ..polyfills.io import write_file_sync
from ..io.summarywriter import summarywriter
from ..io.settings import settings_get


class LSTMTweetClassifier:
	"""Core LSTM-based model to classify tweets."""
	
	def __init__(self, container):
		"""Initialises a new LSTMTweetClassifier."""
		self.settings = settings_get()
		self.container = container
		
		
		self.dir_tensorboard = os.path.join(self.settings.output, "tensorboard")
		self.dir_checkpoints = os.path.join(self.settings.output, "checkpoints")
		self.filepath_tsvlog = os.path.join(self.settings.output, "metrics.tsv")
		self.filepath_summary = os.path.join(self.settings.output, "summary.txt")
		self.filepath_settings = os.path.join(self.settings.output, "settings.json")
		
		
		if not os.path.exists(self.dir_checkpoints):
			os.makedirs(self.dir_checkpoints, 0o750)
		
		
		if not self.container["glove_word_vector_length"]:
			sys.stderr.write("Error: Please initialise the dataset object before initialising the model.\n")
			exit(1)
		
		
		self.make_model()
	
	
	def make_model(self):
		"""Reinitialises the model."""
		# Useful link: https://github.com/fgafarov/learn-neural-networks/blob/master/sequence_classification_LSTM.py
		self.model = tf.keras.Sequential()
		for units in self.settings.model.lstm_units[:-1]:
			logging.info(f"LSTMTweetClassifier: Adding LSTM layer with {units} units")
			self.model.add(tf.keras.layers.LSTM(units, return_sequences=True))
		else:
			logging.info(f"LSTMTweetClassifier: Adding final LSTM layer with {units} units")
			self.model.add(tf.keras.layers.LSTM(units))
		self.model.add(tf.keras.layers.Dense(self.settings.data.categories, activation = "softmax"))
		self.model.compile(
			optimizer="Adam",
			loss="CategoricalCrossentropy",
			metrics=["accuracy"],
			# Raise this to do multiple batches per execution - good for smaller models
			# Unfortunately this requires specifying number of items in the dataset
			steps_per_execution = 1
		)
		self.model.build((
			None,
			self.settings.data.sequence_length,
			self.container["glove_word_vector_length"]
		))
		
		# Write the settings and the model summary to disk
		write_file_sync(self.filepath_settings, self.settings.source)
		summarywriter(self.model, self.filepath_summary)
	
	
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
