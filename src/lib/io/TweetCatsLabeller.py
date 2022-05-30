import sys
import json

from loguru import logger

class TweetCatsLabeller():
	"""Labels tweets using a given categories file."""
	
	def __init__(self, cats):
		self.cats = cats
	
	
	
	def label(self, stream_in, stream_out):
		"""
		Reads tweets from the given input stream, labels them, and then writes
		them to the given output stream.
		"""
		
		i = -1
		for tweet_str in stream_in:
			i += 1
			tweet = json.loads(tweet_str)
			
			tweet["label_cats"] = self.cats.get_category_name(tweet["text"])
			stream_out.write(json.dumps(tweet))
			stream_out.write("\n")
			stream_out.flush()
			
			if i % 1000 == 0:
				sys.stdout.write(f"Labelled {i} tweets\r")
				
		
		print("done!")
	
	
	
