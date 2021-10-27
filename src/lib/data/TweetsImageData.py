import io
import sys
import os
import json
from functools import partial
import numpy

import tensorflow as tf
from loguru import logger

from ..io.settings import settings_get
from ..glove.glove import GloVe
from .CategoryCalculator import CategoryCalculator

# HACK
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
		skipped_missingfile = 0
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
				print(f" [TweetsImageData] stats: {msg}, {skipped} skipped due to missing label, {skipped_missingfile} missing media files")
			
			if next_cat is None:
				continue
			
			
			for media in obj["media"]:
				if media["type"] != "photo":
					continue
				
				filename = os.path.join(
					settings.data.paths.input_media_dir,
					os.path.basename(media["url"])
				)
				
				if not os.path.exists(filename):
					skipped_missingfile = skipped_missingfile + 1
					continue
				
				image = tf.keras.utils.load_img(
					filename,
					target_size=(settings.model.image_size, settings.model.image_size),
					color_mode="rgb"
				)
				
				if settings.model.image_size < 32 and settings.model.type == "resnet":
					image = image.resize((settings.model.image_size, settings.model.image_size))
				
				image = tf.convert_to_tensor(image, dtype=tf.float32)
				
				logger.debug("DEBUG image shape before_transpose "+str(image.shape))
				
				# Convert from channels first ot channels last, since indredibly there isn't actually an option for this
				# I hate Tensorflow for Python so much....
				image = tf.transpose(image, perm=[2,0,1])
				# Ensure we have channels, because apparently the docs for load_img are wrong
				if len(image.shape) < 4:
					image = tf.stack([image, image, image], axis=-1)
				
				
				logger.debug("DEBUG image shape before_preprocess "+str(image.shape))
				
				if settings.model.type == "resnet":
					image = tf.keras.applications.resnet50.preprocess_input(
						image,
						data_format="channels_last"
					)
				else:
					image = image.div(255)
				
				logger.debug("DEBUG image shape after "+str(image.shape))
				yield (
					image,
					tf.one_hot(next_cat, cats.count, dtype="int32")
				)
		
		
		logger.info(f"Epoch end; skipped {skipped} of {i+1} tweets")
		
	
	
	def __new__(cls, filepath_input, container):
		"""
		Returns a new tensorflow dataset object.
		filepath_input (string): The path to the file that contains the input tweets to process.
		container (dict): The container dictionary for passing dynamically computed values around to other parts of the program.
		"""
		global cats
		settings = settings_get()
		cls.init_globals(
			settings.data.paths.categories
		)
		
		dataset = tf.data.Dataset.from_generator(
			partial(cls.generator, filepath_input),
			output_signature=(
				tf.TensorSpec(shape=(
					settings.model.image_size,
					settings.model.image_size,
					3
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
			drop_remainder=True
		)
