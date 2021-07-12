import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
import logging


class ConfusionMatrixMaker:
	"""Makes confusion matrices and renders them to PNG images from pre-trained models."""
	
	def __init__(self, model, cats):
		"""
		Creates new ConfusionMatrixMaker instances.
		model: The Tensorflow model to make predictions with.
		cat_names (string[]): The list of category names (in the SAME order as was used for training).
		"""
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
		for tweet, label in generator:
			ground_truth.append(label)
			
			acc.append(tweet)
			if len(acc) >= self.batch_size:
				predictions_batch = self.model.evaluate(tf.stack(acc))
				for item in predictions_batch:
					predictions.append(item)
				del acc[:]
		
		return ground_truth, predictions
	
	
	def render(self, generator, filepath_output):
		"""
		Renders a confusion matrix using the given dataset.
		dataset (Generator<Tensor, Tensor>): The dataset to use to render the confusion matrix.
		filepath_output (string): The filepath to write the resulting confusion matrix to as a PNG image.
		"""
		
		logging.info("ConfusionMatrixMaker: Starting")
		ground_truth, predictions = self.get_predictions(generator)
		logging.info(f"ConfusionMatrixMaker: Got {len(ground_truth)}, {len(predictions)} ground_truth, predictions")
		
		
		matrix = tf.math.confusion_matrix(ground_truth, predictions)
		logging.into(f"ConfusionMatrixMaker: Got confusion matrix of shape {matrix.shape}")
		
		cat_names = self.cats.get_all_names()
		cats_count = len(cat_names)
		
		plt.clf()
		plt.imshow(
			matrix,
			interpolation="nearest",
			cmap=plt.cm.plasma			# The colour map
		)
		plt.title("Tweet Classifier - Confusion matrix")
		plt.ylabel("Ground Truth")
		plt.xlabel("Prediction")
		ticks = numpy.arange(cats_count)
		plt.xticks(ticks, cat_names, rotation=45)
		plt.yticks(ticks, cat_names)
		
		for i in range(cats_count):
			for j in range(cats_count):
				plt.text(j, i, str(matrix[i][j]))
		
		plt.savefig(filepath_output)
		logging.info(f"ConfusionMatrixMaker: Saved plot to {filepath_output}")
