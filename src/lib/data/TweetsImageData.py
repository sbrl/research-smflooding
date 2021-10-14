import io
import json
from functools import partial
import numpy

import tensorflow as tf

from ..io.settings import settings_get
from ..glove.glove import GloVe
from .CategoryCalculator import CategoryCalculator

# HACK
glove = None
cats = None


class TweetsImageData(tf.data.Dataset):
	"""Converts a LABELLED jsonl tweets file into tensors in the form (image, label)."""
	
	@staticmethod
	def init_globals(filepath_cats):
		global cats
		if cats is None:
			cats = CategoryCalculator(filepath_cats)
	
	@staticmethod
	def generator(filepath_input):
		global cats
		
		settings = settings_get()
		reader = io.open(filepath_input, "r")
		
		skipped = 0
		stats = {}
		i = -1
		for line in reader:
			i += 1
			obj = json.loads(line)
			text = obj["text"].strip()
			
			if "label" not in obj or "media" not in obj:
				skipped += 1
				continue
			
			
			
			next_cat = cats.name2index(obj["label"])
			if next_cat not in stats:
				stats[next_cat] = 0
			stats[next_cat] += 1
			
			# print(f"[DEBUG] next_cat = {next_cat}, total so far = {stats[next_cat]}")
			
			if i > 1 and i % 10000 == 0:
				msg = ""
				for catid in stats:
					msg += f"{cats.index2name(catid)} = {stats[catid]} tweets "
				print(f" [TweetsImageData] stats: {msg}, {skipped} skipped due to missing label or media")
			
			if next_cat is None:
				continue
			
			
			# tf.keras.preprocessing.sequence.pad_sequences doesn't pad correctly :-(
			
			# print(f"[DEBUG] cat=\"{cats.index2name(next_cat)}\" text=\"{glove.tweetvision(text)}\"")
			
			yield (
				tf.constant(
					next_cat,
					dtype="float32"
				),
				tf.one_hot(next_cat, cats.count, dtype="int32")
			)
	
	
	def __new__(cls, filepath_input, container):
		"""
		Returns a new tensorflow dataset object.
		filepath_input (string): The path to the file that contains the input tweets to process.
		container (dict): The container dictionary for passing dynamically computed values around to other parts of the program.
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
		
		return dataset.prefetch(
			tf.data.AUTOTUNE
		).shuffle(
			settings.train.shuffle_buffer
		).batch(
			settings.train.batch_size,
			drop_remainder=settings.model.type == "transformer" # Drop any remainder for transformers because a fixed batch size is required
		)
