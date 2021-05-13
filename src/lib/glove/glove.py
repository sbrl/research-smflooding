#!/usr/bin/env python3
import time
import io
import sys
import logging
import tensorflow as tf

from normalise_text import normalise as normalise_text


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
			self._normalise(str),
			filters = ", \t\n",
			lower = True, split = " "
		)
	
	def _normalise(self, str):
		return normalise_text(str)
	
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


if __name__ == "__main__":
	
	if len(sys.argv) < 3:
		print("""./glove.py
This script handles preprocessing text and convertin it to pretrained GloVe embeddings.
It is intended to be imported as a class, but a CLI is provided for convenience.

Usage:
	path/to/glove.py <gloveloc> <text> [<mode=tweetvision>]

	<gloveloc>	The path to the pretrained GloVe word vectors text file.
	<text>		The text to process.
	<mode>		Optional. The operation mode - defaults to tweetvision, which
				displays what the model will see the text as an array of
				strings. Specify "embeddings" here (without quotes) to display
				the actual word vectors instead.
""")
		exit(0)
	
	_, gloveloc, text = sys.argv
	mode = sys.argv[3] if len(sys.argv) >= 4 else "tweetvision"
	
	glove = GloVe(gloveloc)
	
	result = glove._tokenise(text) if mode == "tweetvision" else glove.embeddings(text)
	
	print()
	print(result)
