import json

import tensorflow as tf

class TweetLabeller():
	"""Labels tweets using a given (pre-initialised) TweetClassifier instance."""
	
	def __init__(self, model, cats, batch_size=32):
		self.model = model
		self.cats = cats
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
				"tensor": self.tweet2tensor(tweet)
			})
			if len(acc) >= self.batch_size:
				stacked = tf.stack([ item.tensor for item in acc ])
				print("STACKED_SHAPE", stacked.shape)
				predictions_batch = self.model.predict_class_ids(stacked, self.batch_size)
				print("PREDICTIONS_BATCH", predictions_batch)
				# Process the predictions
				# for index in range(0, len(predictions_batch)):
				for batch_index in range(0, len(predictions_batch)):
					label_index = predictions_batch[batch_index]
					print("ITEM", label_index)
					if label_index is not None:
						acc[batch_index].obj.label = self.cats.get_category_name(label_index)
						stream_out.write(json.dumps(acc[batch_index].obj))
						stream_out.write("\n")
				
				# Empty the accumulators
				del acc[:]
	
		
	
