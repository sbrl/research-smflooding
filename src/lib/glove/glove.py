import time
import io
import logging
import tensorflow as tf
import re				# Regex
import inspect


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
		
		self.filepath = filepath
		self.load()
	
	def load(self):
		"""Loads the GloVe database from a given file."""
		start = time.time()
		handle = io.open(self.filepath, "r")
		for i, line in enumerate(handle):
			parts = line.split(" ", maxsplit=1)
			key = parts[0].lstrip("<").rstrip(">")
			
			self.data[key] = list(map(
				lambda el: float(el),
				parts[1].split(" ")
			))
			if i % 10000 == 0:
				print(f"Loading GloVe from '{self.filepath}': {i}", end="\r")
		
		handle.close()
		logging.info(f"GloVe: Loaded embeddings in {round(time.time() - start, 3)}s.")
	
	def lookup(self, token):
		"""Looks up the given token in the loaded embeddings."""
		if token not in self.data:
			return None
		
		return self.data[token]
		
	
	def _tokenise(self, str):
		return tf.keras.preprocessing.text.text_to_word_sequence(
			str,
			filters = ", \t\n",
			lower = True, split = " "
		)
	
	def _normalise(str):
		# TODO: Adapt https://gist.github.com/tokestermw/cb87a97113da12acb388
		
		raise NotImplementedError("TODO: Finish this")
		
		return result
	
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
	
	def embeddings(self, str):
		"""Converts the given string to a list of word embeddings."""
		result = []
		# TODO: Handle out-of-vocabulary words better than just stripping them
		for i, token in enumerate(self._tokenise(str)):
			embedding = self.lookup(token)
			if embedding is None:
				logging.debug(f"[DEBUG] {token} was none")
				continue
			
			result.append(embedding)
		
		return result
