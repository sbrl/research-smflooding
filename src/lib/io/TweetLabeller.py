import logging
import json

import tensorflow as tf

class TweetLabeller():
	"""Labels tweets using a given (pre-initialised) TweetClassifier instance."""
	
	def __init__(self, model, glove, cats, batch_size=32):
		self.model = model
		self.cats = cats
		self.glove = glove
		self.batch_size = batch_size
		
		# This is just silly....
		# self.model.model.get_config()["layers"][0]["config"]["batch_input_shape"] should look something like (None, 100, 200)
		self.sequence_length = self.model.model.get_config()["layers"][0]["config"]["batch_input_shape"][1]
	
	
	def tweet2tensor(self, tweet_text):
		"""
		Converts a tweet to a tensor using GloVe.
		Respects the sequence length of the model provided at the initialisation of this TweetLabeller instance.
		tweet_text (string): The string to convert.
		"""
		return self.glove.embeddings(
			self.cats.strip_markers(tweet_text),
			self.sequence_length
		)
	
	
	def label(self, stream_in, stream_out):
		"""
		Reads tweets from the given input stream, labels them, and then writes
		them to the given output stream.
		"""
		
		acc = []
		for tweet_str in stream_in:
			tweet = json.loads(tweet_str)
			
			acc.append({
				"obj": tweet,
				"tensor": self.tweet2tensor(tweet["text"])
			})
			if len(acc) >= self.batch_size:
				stacked = tf.stack([ item["tensor"] for item in acc ])
				logging.info(f"STACKED_SHAPE {stacked.shape}")
				predictions_batch = self.model.predict_class_ids(stacked, self.batch_size)
				logging.info(f"PREDICTIONS_BATCH {predictions_batch}")
				# Process the predictions
				# for index in range(0, len(predictions_batch)):
				for batch_index in range(0, len(predictions_batch)):
					label_index = predictions_batch[batch_index]
					print("ITEM", label_index)
					if label_index is not None:
						acc[batch_index]["obj"]["label"] = self.cats.index2name(label_index)
						stream_out.write(json.dumps(acc[batch_index]["obj"]))
						stream_out.write("\n")
				
				# Empty the accumulators
				del acc[:]
	
	
	
