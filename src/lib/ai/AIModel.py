import os
import io
import sys
from abc import abstractmethod

import logging

from ..polyfills.io import write_file_sync
from ..io.summarywriter import summarywriter, summarylogger
from ..io.settings import settings_get
import tensorflow as tf

class AIModel:
	"""Abstract base class for managing AI models."""
	def __init__(self, container, filepath_checkpoint):
		self.container = container
		self.filepath_checkpoint = filepath_checkpoint
	
	def setup(self):
		if self.filepath_checkpoint is None:
			logging.info("AIModel: Creating new model")
			self.settings = settings_get()
			
			self.dir_tensorboard = os.path.join(self.settings.output, "tensorboard")
			self.dir_checkpoints = os.path.join(self.settings.output, "checkpoints")
			self.filepath_tsvlog = os.path.join(self.settings.output, "metrics.tsv")
			self.filepath_summary = os.path.join(self.settings.output, "summary.txt")
			self.filepath_settings = os.path.join(self.settings.output, "settings.toml")
			
			if not os.path.exists(self.dir_checkpoints):
				os.makedirs(self.dir_checkpoints, 0o750)
			
			
			self.model = self.make_model()
			
			
			# Write the settings and the model summary to disk
			write_file_sync(self.filepath_settings, self.settings.source)
			summarywriter(self.model, self.filepath_summary)
			summarylogger(self.model)
			
			logging.info("Model summary above")
			
		else:
			logging.info(f"ImageClassifier: Loading checkpoint from {self.filepath_checkpoint}")
			self.load_model(self.filepath_checkpoint)
	
	@abstractmethod
	def make_model(self):
		"""Creates and returns a new model instance."""
		pass
	
	@abstractmethod
	def custom_layers(self):
		"""
		Returns a list of custom layers used.
		This is important for loading models back in again!
		"""
		pass
	
	def load_model(self, filepath_checkpoint):
		"""
		Loads a saved model from the given filename.
		filepath_checkpoint (string): The filepath to load the saved model from.
		"""
		
		if not os.path.exists(filepath_checkpoint):
			print(f"AIModel Error: No such file or directory {filepath_checkpoint}")
			sys.exit(2)
		
		
		self.model = tf.keras.models.load_model(filepath_checkpoint, custom_objects=self.custom_layers())
		
	
	def make_callbacks(self):
		"""Generates a list of callbacks to be called when a model is training."""
		return [
			tf.keras.callbacks.ModelCheckpoint(
				filepath=os.path.join(
					self.dir_checkpoints,
					"checkpoint_e{epoch:d}_val_acc{val_accuracy:.3f}.hdf5"
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
	
	def predict(self, data, batch_size=None):
		"""Makes a prediction for the given input data with the AI model, but does not update any weights."""
		return self.model.predict(
			data,
			batch_size=batch_size
		)
	
