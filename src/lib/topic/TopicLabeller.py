import sys
import json
from pprint import pprint

from loguru import logger

import tensorflow as tf

class TopicLabeller():
	"""Labels tweets using a given (pre-initialised) TopicAnalyser instance."""
	
	def __init__(self, model, batch_size=512):
		self.model = model
		self.batch_size = batch_size
	
	def top_topic(self, prediction):
		top_i = float("-inf")
		top_prediction = float("-inf")
		for i, value in prediction:
			if value > top_prediction:
				top_i = i
				top_prediction = value
		
		return top_i, top_prediction
	
	def label(self, stream_in, stream_out):
		"""
		Reads tweets from the given input stream, labels them, and then writes
		them to the given output stream.
		"""
		
		i = 0
		for tweet_str in stream_in:
			tweet = json.loads(tweet_str)
			
			prediction = self.model.predict(tweet["text"])[0]
			topic_id, confidence = self.top_topic(prediction)
			tweet["label_topic"] = int(topic_id)
			tweet["label_topic_confidence"] = round(float(confidence), 4)
			stream_out.write(json.dumps(tweet))
			stream_out.write("\n")
			stream_out.flush()
			
			if i % 10000 == 0:
				sys.stderr.write(f"{i} tweets processed\r")
			i = i+1
	
