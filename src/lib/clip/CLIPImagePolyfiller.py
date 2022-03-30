import io
import json
import random
import time
import sys
import os
import subprocess

from loguru import logger
import torch
import clip

def clear_line():
    sys.stderr.write("{}\r".format(' '*os.get_terminal_size().columns))

class CLIPImagePolyfiller(object):
	def __init__(self, dataset_images, clip_model, device, cats=None, batch_size=64, use_tensor_cache=True):
		super(CLIPImagePolyfiller, self).__init__()
		
		self.use_tensor_cache = use_tensor_cache
		self.device = device
		self.batch_size = batch_size
		self.candidate_threshold = 0.9
		
		self.cats = cats
		self.dataset = dataset_images
		
		self.data = torch.utils.data.DataLoader(
			dataset_images,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=os.cpu_count()
		)
		
		self.clip_model = clip_model
		
		self.tensor_cache = {}
		if self.use_tensor_cache:
			self.prefill_cache()
	
	def prefill_cache(self):
		logger.info(f"Prefilling image tensor cache.")
		
		time_start = time.time()
		time_last_update = time.time()
		for step, image_batch in enumerate(self.data):
			self.tensor_cache[step] = self.encode_image_batch(image_batch)
			
			time_current = time.time()
			if step == 0 or time_current - time_last_update > 2:
				elapsed = time_current - time_start
				percent = round(((step*self.batch_size)/self.dataset.length)*100, 2)
				eta = elapsed/(step*self.batch_size) * (self.dataset.length - step)
				sys.stdout.write(f"Prefill tensor cache: {step} / {self.dataset.length} ({percent}%) | Time: {elapsed}s > {eta}\r")
		
		logger.info(f"Tensor cache filled in {round(time.time() - time_start, 2)}s.")
	
	
	def encode_image_batch(self, image_batch):
		image_features = self.clip_model.encode_image(image_batch.to(self.device))
		image_features /= image_features.norm(dim=-1, keepdim=True)
		return image_features
	
	
	def label(self, filepath_input, filepath_output):
		handle_in = io.open(filepath_input, "r")
		handle_out = io.open(filepath_output, "w")
		
		i = 0
		while True:
			line = handle_in.readline()
			if line is None or line == "":
				break
			i = i + 1
			try:
				obj = json.loads(line)
			except Exception as error:
				logger.warning(f"Error while parsing JSON on line {i}, skipping: {error}")
				continue
			
			if i % 1000 == 0:
				logger.info(f"Labelled {i} tweets")
			
			if "media" in obj:
				continue
			
			text = obj["text"].strip()
			
			# If the tweet text doesn't contain any supported emojis, then don't bother either
			if self.cats and self.cats.get_category_index(text) is None:
				obj["media_clip"] = None
				obj["media_clip_confidence"] = -1
				
				handle_out.write(json.dumps(obj) + "\n")
				if i % 100:
					handle_out.flush()
				continue
			
			with torch.no_grad():
				time_start = time.time()
				
				text = self.clip_model.encode_text(
					clip.tokenize(text, truncate=True).to(self.device)
				).to(self.device)
				text_features = torch.stack([text]*self.batch_size, 0).squeeze(1)
				
				text_features /= text_features.norm(dim=-1, keepdim=True)
				logger.info(f"text_features.shape {text_features.shape}")
				
				candidates = []
				image_id_best = -1		# The id of the one we're most confident about
				image_confidence = -1	# How confident we are in the current best
				
				time_step = time.time()
				time_dataset = 0
				enumerator = None
				if self.use_tensor_cache:
					enumerator = self.tensor_cache.items()
				else:
					enumerator = enumerate(self.data)
				for step, image_batch in enumerator:
					time_dataset += time.time() - time_step
					image_features = self.encode_image_batch(image_batch)
					
					similarity = (100 * torch.matmul(text_features, image_features.T)).softmax(dim=-1)
					similarity = similarity[0]
					
					for similarity_i, value in enumerate(similarity):
						if value > image_confidence:
							image_id_best = self.batch_size * step + similarity_i
							image_confidence = value
						if value > self.candidate_threshold:
							candidates.append([self.batch_size * step + similarity_i, value])
					
					percent = ((step*self.batch_size)/self.dataset.length)*100
					percent_dataset = (time_dataset/(time.time()-time_start))*100
					clear_line()
					sys.stdout.write(f"STEP {i}:{step}, {step*self.batch_size}/{self.dataset.length} ({round(percent, 2)}%) id_best: {image_id_best} filename: {self.dataset.get_filename(image_id_best)} confidence: {image_confidence.item()} candidates: {len(candidates)}, {round(percent_dataset, 2)}% dataset overhead\r")
					
					time_step = time.time()
				
				selected_id = image_id_best
				selected_confidence = image_confidence
				# If we found at least 1 item above the confidence threshold, randomly select one to avoid a consistent high scorer from being picked every time
				if candidates:
					picked = random.choice(candidates)
					selected_id = picked[0]
					selected_confidence = picked[1]
				
				obj["media_clip"] = os.path.basename(self.dataset.get_filename(selected_id))
				obj["media_clip_confidence"] = selected_confidence.item()
				print("DEBUG media_clip", obj["media_clip"])
				print("DEBUG media_clip_confidence", obj["media_clip_confidence"])
				logger.info(f"Tweet {i}: selected {selected_id} → "+obj["media_clip"]+f" confidence "+str(obj["media_clip_confidence"])+f" in {round(time.time() - time_start, 2)}s")
				
				handle_out.write(json.dumps(obj) + "\n")
				if i % 100:
					handle_out.flush()
			
