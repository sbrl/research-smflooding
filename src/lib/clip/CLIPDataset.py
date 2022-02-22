import os
import io
import json

import torch
from PIL import Image
import clip


class CLIPDataset(torch.utils.data.IterableDataset):
	def __init__(self, filepath_tweets, dir_media, cats, batch_size=64, device="cpu", clip_model_name="ViT-B/32", **kwargs):
		super(CLIPDataset).__init__()
		
		
		self.clip_model_name = clip_model_name
		self.filepath_tweets = filepath_tweets
		self.device = device
		self.dir_media = dir_media
		self.batch_size = batch_size
		
		self.cats = cats
		_, self.preprocess = clip.load(self.clip_model_name, device=device)
		self.clip_sequence_length = clip.tokenize.__defaults__[0]
		if type(self.clip_sequence_length) != int:
			self.clip_sequence_length = 77
		
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
		
		self.handle_tweets = io.open(self.filepath_tweets, "r")
		
		self.queue = []
		print("DEBUG THIS IS __iter__")
		return self
	
	
	def ____get_next_item(self):
		if not self.queue:
			self.add_to_queue()
			
			if not self.queue:
				return None
		
		return self.queue.pop(0)
	
	def __next__(self):
		acc = []
		for i in range(0, self.batch_size):
			next_item = self.____get_next_item()
			
			if next_item is None:
				raise StopIteration
			acc.append(next_item)
		
		batched = self.do_collate(acc)
		return batched
	
	
	def do_collate(self, items):
		return {
			"labels": torch.stack([ torch.as_tensor(item["label"], dtype=torch.float32) for item in items]),
			"text": torch.stack([ item["text"] for item in items]).int(),
			"images": torch.stack([ item["image"] for item in items])
		}
	
	def add_to_queue(self):
		while True:
			line = self.handle_tweets.readline()
			if line is None:
				return # We've reach the end of the output
			obj = json.loads(line)
			text = obj["text"].strip()
			
			if "media" not in obj:
				continue
			
			cat = self.cats.get_category_index(text)
			
			if cat is None:
				continue
			
			tweet_text = clip.tokenize(text, truncate=True).squeeze(0).to(self.device)
			
			added = 0
			for media in obj["media"]:
				if media["type"] != "photo":
					continue
				
				filename = os.path.join(
					self.dir_media,
					os.path.basename(media["url"])
				)
				
				if not os.path.exists(filename):
					continue
				
				image = self.preprocess(Image.open(filename)).unsqueeze(0).to(self.device)
				
				self.queue.append({
					"label": cat,
					"text": tweet_text,
					"image": image
				})
				added += 1
			
			
			if added > 0:
				break
