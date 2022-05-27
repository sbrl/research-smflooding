import time
import io
import sys
from loguru import logger
import numpy
import tensorflow as tf
from ..polyfills.string import removeprefix, removesuffix

from .normalise_text import normalise as normalise_text


class GloVe:
	"""
	Manages pre-trained GloVe word vectors.
	Ref https://www.damienpontifex.com/posts/using-pre-trained-glove-embeddings-in-tensorflow/
	Download pre-trained word vectors from here: https://nlp.stanford.edu/projects/glove/
	"""
	
	def __init__(self, filepath):
		"""
		Initialises a new GloVe class instance.
		filepath (string): The path to the file to load the pre-trained GloVe embeddings from.
		"""
		super(GloVe, self).__init__()
		
		self.data = {}
		
		self.word_length = None
		self.filepath = filepath
		self.load()
	
	def load(self):
		"""Loads the GloVe database from a given file."""
		sys.stderr.write("\n")
		start = time.time()
		handle = io.open(self.filepath, "r")
		for i, line in enumerate(handle):
			parts = line.split(" ", maxsplit=1)
			
			# We do NOT strip < and > here, because we do a lookup later on that.
			self.data[parts[0]] = list(map(
				lambda el: float(el),
				parts[1].split(" ")
			))
			if self.word_length is None:
				self.word_length = len(self.data[parts[0]])
			
			# Update the CLI
			if i % 10000 == 0:
				sys.stderr.write(f"\rLoading GloVe from '{self.filepath}': {i}...")
		
		handle.close()
		sys.stderr.write(f" done in {round(time.time() - start, 3)}s.\n")
	
	def lookup(self, token: str):
		"""Looks up the given token in the loaded embeddings."""
		key = token
		
		if key not in self.data:
			key = self.strip_outer(token)	# Try removing < and >
		if key not in self.data:
			key = f"<{token}>"				# Try wrapping in < and >
		if key not in self.data:
			return None						# Give up
		
		return self.data[key]				# We found it!
	
	def strip_outer(self, str: str) -> str:
		"""Strips < and > from the given input string."""
		return removesuffix(removeprefix(str, "<"), ">")
	
	def _tokenise(self, str: str):
		"""Splits the input string into tokens using Keras."""
		return tf.keras.preprocessing.text.text_to_word_sequence(
			self._normalise(str),
			filters = ", \t\n",
			lower = True, split = " "
		)
	
	def _normalise(self, str):
		"""Normalises input text to be suitable to GloVe lookup."""
		return normalise_text(str)
	
	
	###########################################################################
	
	
	def word_vector_length(self):
		"""Returns the length of a single word vector."""
		return self.word_length
	
	
	def tweetvision(self, str):
		"""
		Convert a string to a list of tokens as the AI will see it.
		Basically  the same as .embeddings(str), but returns the tokens instead of the embeddings.
		"""
		result = []
		for i, token in enumerate(self._tokenise(str)):
			if self.lookup(token) is None:
				continue
			else:
				result.append(token)
		
		return result
	
	def embeddings(self, str, length=-1):
		"""
		Converts the given string to a list of word embeddings.
		str (string): The string to convert to an embedding.
		length (number): The number of tokens that the returned embedding should have. -1 (the default value) indicates that no length normalisation should be performed.
		"""
		result = []
		# TODO: Handle out-of-vocabulary words better than just stripping them
		for i, token in enumerate(self._tokenise(str)):
			embedding = self.lookup(token)
			if embedding is None:
				# logger.debug(f"[DEBUG] {token} was none")
				continue
			
			result.append(embedding)
		
		# Normalise the embedding length if we're asked to
		if length > -1:
			result = result[-length:]
			
			shortfall = length - len(result)
			for _ in range(shortfall):
				result.append(numpy.zeros(self.word_vector_length()))
		
		return result
