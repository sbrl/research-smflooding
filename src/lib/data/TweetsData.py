import io
import json

import tensorflow as tf

from ..io.settings import settings_get
from ..glove.glove import GloVe
from .CategoryCalculator import CategoryCalculator

# HACK
glove = None
cats = None


class TweetsData(tf.data.Dataset):
	"""Converts a jsonl tweets file into tensors."""
	
	def _generator(filepath_input):
		global glove, cats
		
		settings = settings_get()
		reader = io.open(filepath_input, "r")
		
		for line in reader:
			obj = json.loads(line)
			text = obj["text"].strip()
			
			next_cats = cats.get_category_name(text)
			
			if not next_cats:
				continue
			
			yield (
				tf.constant(
					tf.keras.preprocessing.sequence.pad_sequences(
						glove.embeddings(text),
						dtype="float32",
						padding="post",
						maxlen=settings.data.sequence_length
					),
					dtype="float32"
				),
				tf.constant(
					next_cats,
					dtype="float32"
				)
			)
	
	
	def __new__(this_class, filepath_input):
		"""Returns a new tensorflow dataset object."""
		global glove, cats
		settings = settings_get()
		cats = CategoryCalculator()
		
		# Globalise the instance
		if glove is None:
			glove = GloVe(settings.data.paths.glove)
		
		
		return tf.data.Dataset.from_generator(
			this_class._generator,
			output_signature=(
				tf.TensorSpec(shape=(
					settings.data.sequence_length,
					glove.word_vector_length()
				), dtype="float32")
			),
			args = ( filepath_input )
		).batch(settings.train.batch_size).prefetch(tf.data.AUTOTUNE)
