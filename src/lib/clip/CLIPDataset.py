import os
import io
import json

from loguru import logger
import torch
from PIL import Image
import clip


class CLIPDataset(torch.utils.data.IterableDataset):
	def __init__(self, filepath_tweets, dir_media, cats, clip_preprocess, batch_size=64, device="cpu", clip_label_threshold=0.75, do_images=True, **kwargs):
		super(CLIPDataset).__init__()
		
		self.filepath_tweets = filepath_tweets
		self.device = device
		self.dir_media = dir_media
		self.batch_size = batch_size
		self.do_images = do_images
		
		self.clip_label_threshold = clip_label_threshold
		
		logger.info(f"START with:\n* filepath_tweets {self.filepath_tweets}\n* device {self.device}\n* dir_media {self.dir_media}\n* batch_size {self.batch_size}\n* do_images {self.do_images}\n* clip_label_threshold {self.clip_label_threshold}")

		logger.info(f"START with:\n* filepath_tweets {self.filepath_tweets}\n* batch_size {self.batch_size}\n* cats {self.cats}\n* clip_sequence_length {self.clip_sequence_length}")		
		self.cats = cats
		self.preprocess = clip_preprocess
		self.clip_sequence_length = clip.tokenize.__defaults__[0]
		if type(self.clip_sequence_length) != int:
			self.clip_sequence_length = 77
		
		###
		## State-specific member variables
		###
		self.handle_tweets = None
		self.queue = []
		
		self.count_total = 0
		self.count_withmedia = 0
		self.count_withemoji = 0
		
		if not self.do_images:
			self.image_blank = Image.new("RGB", (224, 224), (225, 225, 225))
	
	
	def __iter__(self):
		# Called once at the beginning to reset state
		if self.handle_tweets is not None:
			self.handle_tweets.close()
			self.handle_tweets = None
		
		logger.info(f"previous | total: {self.count_total} → withmedia: {self.count_withmedia} → withemoji: {self.count_withemoji}")
		self.count_total = 0
		self.count_withmedia = 0
		self.count_withemoji = 0
		
		self.handle_tweets = io.open(self.filepath_tweets, "r", errors="replace")
		
		self.queue = []
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
			"labels": torch.stack([ torch.as_tensor(item["label"], dtype=torch.long) for item in items]),
			"text": torch.stack([ item["text"] for item in items]).int().to(self.device),
			"images": torch.stack([ item["image"] for item in items])
		}
	
	def add_to_queue(self):
		while True:
			try:
				self.count_total += 1
				line = self.handle_tweets.readline()
				if line == "":
					break
				if line is None:
					logger.info(f"total: {count_total} → withmedia: {count_withmedia} → withemoji: {count_withemoji}")
					return # We've reach the end of the output
				obj = json.loads(line)
			except Exception as error:
				logger.warning(f"Error while reading or parsing line {self.count_total}, skipping: {error}")
				continue
			
			text = obj["text"].strip()
			
			
			media_items = []
			if "media" in obj:
				media_items.extend(obj["media"])
			
			# If there's no media and we labelled it with CLIP, use that instead
			if "media" not in obj and "media_clip" in obj and obj["media_clip"] is not None and obj["media_clip_confidence"] is not None and  obj["media_clip_confidence"] >= self.clip_label_threshold:
				media_items.append({
					"type": "photo",
					"url": obj["media_clip"]
				})
			
			if not media_items:
				continue
			
			self.count_withmedia += 1
			
			cat = self.cats.get_category_index(text)
			
			if cat is None:
				continue
			
			self.count_withemoji += 1
			
			cat = torch.as_tensor(cat, dtype=torch.int).to(self.device)
			
			tweet_text = clip.tokenize(text, truncate=True).squeeze(0).to(self.device)
			
			added = 0
			for media in media_items:
				if media["type"] != "photo":
					continue
				
				filename = os.path.join(
					self.dir_media,
					os.path.basename(media["url"])
				)
				
				if not os.path.exists(filename):
					continue
				
				if self.do_images:
					image = Image.open(filename)
				else:
					image = self.image_blank
				
				image = self.preprocess(image).to(self.device)
				
				self.queue.append({
					"label": cat.to(device=self.device),
					"text": tweet_text.to(device=self.device),
					"image": image.to(device=self.device)
				})
				added += 1
			
			
			if added > 0:
				break
