import io
import json

import torch
from PIL import Image
import clip


class CLIPDataset(torch.utils.data.IterableDataset):
	def __init__(self, filepath_tweets, dir_media, cats, device="cpu", clip_model_name="ViT-B/32", **kwargs):
		super(CLIPDataset).__init__()
		
		
		self.clip_model_name = clip_model_name
		self.filepath_tweets = filepath_tweets
		self.device = device
		self.dir_media = dir_media
		
		self.cats = cats
		_, self.preprocess = clip.load(self.clip_model_name, device=device)
		
		###
		## State-specific member variables
		###
		self.handle_tweets = None
		self.queue = []
		
	def __iter__(self):
		# Called once at the beginning to reset state
		if self.handle_tweets is not None:
			self.handle_tweets.close()
			self.handle_tweets = None
		
		self.handle_tweets = io.open(self.filepath_tweets)
		
		self.queue = []
		
		return self
	
	
	def __next__(self):
		if not self.queue:
			self.add_to_queue()
			
			if not self.queue:
				return None
		
		return self.queue.pop(0)
	
	
	def add_to_queue(self):
		while True:
			line = self.handle_tweets.readline()
			if line is None:
				return
			obj = json.loads(line)
			text = obj["text"].strip()
			
			if "media" not in obj:
				continue
			
			cat = self.cats.get_category_index(text)
			
			if cat is None:
				continue
			
			next_cat = cats.get_category_index(text)
			tweet_text = clip.tokenize([ text ])[0].to(device)
			
			
			for media in obj["media"]:
				filename = os.path.join(
					self.dir_media,
					os.path.basename(media["url"])
				)
				
				if not os.path.exists(filename):
					continue
				
				image = self.preprocess(Image.open(filename)).unsqueeze(0).to(self.device)
				
				self.queue.append((tweet_text, image))
			
			break
			
