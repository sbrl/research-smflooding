from loguru import logger
import json

import tensorflow as tf

class TweetLabeller():
	"""Labels tweets using a given (pre-initialised) TopicAnalyser instance."""
	
	def __init__(self, model, batch_size=512):
		self.model = model
		self.batch_size = batch_size
	
	
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
				strings = [ item.obj.text ]
				predictions_batch = self.model.predict(strings)
				# Process the predictions
				# for index in range(0, len(predictions_batch)):
				for batch_index in range(0, len(predictions_batch)):
					label_index = predictions_batch[batch_index]
					if label_index is not None:
						acc[batch_index]["obj"]["label_topic"] = self.cats.index2name(label_index)
						stream_out.write(json.dumps(acc[batch_index]["obj"]))
						stream_out.write("\n")
						stream_out.flush()
				
				# Empty the accumulators
				del acc[:]
	
	
	
