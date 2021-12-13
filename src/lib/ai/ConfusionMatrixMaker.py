import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
from loguru import logger
import os


class ConfusionMatrixMaker:
	"""Makes confusion matrices and renders them to PNG images from pre-trained models."""
	
	def __init__(self, model, cats, min_confidence):
		"""
		Creates new ConfusionMatrixMaker instances.
		model: The Tensorflow model to make predictions with.
		cat_names (string[]): The list of category names (in the SAME order as was used for training).
		min_confidence (number): The minimum confidence required by the model to include it in confusion matrices. Should be a floating-point number between 0 (no confidence) and 1 (full confidence).
		"""
		self.min_confidence = min_confidence
		self.model = model
		self.cats = cats
		
		self.batch_size = 32
	
	
	def get_predictions(self, generator):
		"""
		Returns 2 lists of ground truth and predicted values.
		generator (Generator<Tensor, Tensor>): The generator that generates the data to consume.
		"""
		ground_truth = []
		predictions = []
		
		acc = []
		acc_truth =  []
		for tweet, label in generator:
			label_index = -1
			for i in range(0, len(label)):
				if label[i] == 1:
					label_index = i
			
			if label_index == -1:
				raise Exception(f"Error: One-hot encoded label '{label}' did not have at least one element with a value of 1.")
			
			acc_truth.append(label_index)
			acc.append(tweet)
			if len(acc) >= self.batch_size:
				stacked = tf.stack(acc)
				print("STACKED_SHAPE", stacked.shape)
				predictions_batch = self.model.predict_class_ids(stacked, self.batch_size)
				print("PREDICTIONS_BATCH", predictions_batch)
				# Process the predictions
				for index in range(0, len(predictions_batch)):
					item = predictions_batch[index]
					print("ITEM", item)
					print("LABEL", acc_truth[index])
					if item is not None:
						predictions.append(item)
						ground_truth.append(acc_truth[index])
				
				# Empty the accumulators
				del acc[:]
				del acc_truth[:]
		
		return ground_truth, predictions
	
	def render(self, generator, filepath_output):
		"""
		Renders a confusion matrix using the given dataset.
		dataset (Generator<Tensor, Tensor>): The dataset to use to render the confusion matrix.
		filepath_output (string): The filepath to write the resulting confusion matrix to as a PNG image.
		"""
		
		logger.info("ConfusionMatrixMaker: Starting")
		ground_truth, predictions = self.get_predictions(generator)
		logger.info(f"ConfusionMatrixMaker: Got {len(ground_truth)}, {len(predictions)} ground_truth, predictions")
		
		
		matrix = tf.math.confusion_matrix(ground_truth, predictions)
		logger.info(f"ConfusionMatrixMaker: Got confusion matrix of shape {matrix.shape}")
		
		cat_names = self.cats.get_all_names()
		cats_count = len(cat_names)
		
		plt.clf()
		plt.imshow(
			matrix,
			interpolation="nearest",
			cmap=plt.cm.plasma			# The colour map
		)
		title_code = os.path.basename(filepath_output).replace(".png", "")
		plt.title(f"Tweet Classifier - Confusion matrix\n{title_code}")
		plt.ylabel("Ground Truth")
		plt.xlabel("Prediction")
		ticks = numpy.arange(cats_count)
		plt.xticks(ticks, cat_names, rotation=45)
		plt.yticks(ticks, cat_names)
		
		for i in range(cats_count):
			for j in range(cats_count):
				plt.text(j, i, str(matrix[i][j].numpy()), backgroundcolor=(1, 1, 1, 0.5))
		
		plt.tight_layout()
		plt.savefig(filepath_output)
		logger.info(f"ConfusionMatrixMaker: Saved plot to {filepath_output}")
