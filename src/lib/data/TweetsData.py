import io
import json
from functools import partial
import numpy

import tensorflow as tf

from ..io.settings import settings_get
from ..glove.glove import GloVe
from .CategoryCalculator import CategoryCalculator
from .Smoteifier import smoteify

# HACK
glove = None
cats = None


class TweetsData(tf.data.Dataset):
	"""Converts a jsonl tweets file into tensors."""
	
	@staticmethod
	def init_globals(filepath_cats, filepath_glove):
		global glove, cats
		if cats is None:
			cats = CategoryCalculator(filepath_cats)
		# Globalise the instance
		if glove is None:
			glove = GloVe(filepath_glove)
	
	@staticmethod
	def generator(filepath_input):
		global glove, cats
		
		settings = settings_get()
		reader = io.open(filepath_input, "r")
		
		stats = {}
		i = -1
		for line in reader:
			i += 1
			obj = json.loads(line)
			text = obj["text"].strip()
			
			next_cat = cats.get_category_index(text)
			text_stripped = cats.strip_markers(text)
			if next_cat not in stats:
				stats[next_cat] = 0
			stats[next_cat] += 1
			
			# print(f"[DEBUG] next_cat = {next_cat}, total so far = {stats[next_cat]}")
			
			if i > 1 and i % 10000 == 0:
				msg = ""
				for catid in stats:
					msg += f"{cats.index2name(catid)} = {stats[catid]} tweets "
				print(f" [TweetsData] stats: {msg}")
			
			if next_cat is None:
				continue
			
			text_glove = glove.embeddings(text_stripped)
			
			# Ensure we don't have too many tokens in the sequence
			if len(text_glove) > settings.data.sequence_length:
				text_glove = text_glove[-settings.data.sequence_length:]
			
			# Make up any shortfall with zeroes at the end
			shortfall = settings.data.sequence_length - len(text_glove)
			for _ in range(shortfall):
				text_glove.append(numpy.zeros(glove.word_vector_length()))
			
			# tf.keras.preprocessing.sequence.pad_sequences doesn't pad correctly :-(
			
			# print(f"[DEBUG] cat=\"{cats.index2name(next_cat)}\" text=\"{glove.tweetvision(text)}\"")
			
			yield (
				tf.constant(
					text_glove,
					dtype="float32"
				),
				tf.one_hot(next_cat, cats.count, dtype="int32")
			)
	
	
	def __new__(cls, filepath_input, container, smote = False):
		"""
		Returns a new tensorflow dataset object.
		filepath_input (string): The path to the file that contains the input tweets to process.
		container (dict): The container dictionary for passing dynamically computed values around to other parts of the program.
		smote (bool): Whether to activate the SMOTEIFICATION system
		"""
		global glove, cats
		settings = settings_get()
		cls.init_globals(
			settings.data.paths.categories,
			settings.data.paths.glove
		)
		
		container["glove_word_vector_length"] = glove.word_vector_length()
		
		dataset = tf.data.Dataset.from_generator(
			partial(cls.generator, filepath_input),
			output_signature=(
				tf.TensorSpec(shape=(
					settings.data.sequence_length,
					glove.word_vector_length()
				), dtype="float32"),
				tf.TensorSpec(
					shape=(
						cats.count
					),
					dtype="int32"
				)
			)
		)
		
		if smote is True:
			print("Activating SMOTEIFICATION system")
			dataset = smoteify(dataset)
		
		return dataset.prefetch(
			tf.data.AUTOTUNE
		).shuffle(
			settings.train.shuffle_buffer
		).batch(
			settings.train.batch_size,
			drop_remainder=settings.model.type == "transformer" # Drop any remainder for transformers because a fixed batch size is required
		)
