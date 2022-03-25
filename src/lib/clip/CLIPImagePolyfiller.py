import io
import json
import random
import time
import sys
import os

from loguru import logger
import torch
import clip

class CLIPImagePolyfiller(object):
	def __init__(self, dataset_images, clip_model, device, batch_size=64):
		super(CLIPImagePolyfiller, self).__init__()
		
		self.device = device
		self.batch_size = batch_size
		self.candidate_threshold = 0.9
		
		self.dataset = dataset_images
		self.data = torch.utils.data.DataLoader(
			dataset_images,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=os.cpu_count()
		)
		
		self.clip_model = clip_model
	
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
			
			# TODO: If the tweet text doesn't contain any supported emojis, then don't bother either
			
			with torch.no_grad():
				time_start = time.time()
				
				text = self.clip_model.encode_text(
					clip.tokenize(obj["text"].strip(), truncate=True).to(self.device)
				).to(self.device)
				text_features = torch.stack([text]*self.batch_size, 0).squeeze(1)
				
				text_features /= text_features.norm(dim=-1, keepdim=True)
				logger.info(f"text_features.shape {text_features.shape}")
				
				candidates = []
				image_id_best = -1		# The id of the one we're most confident about
				image_confidence = -1	# How confident we are in the current best
				
				for step, image_batch in enumerate(self.data):
					image_features = self.clip_model.encode_image(image_batch)
					image_features /= image_features.norm(dim=-1, keepdim=True)
					
					similarity = (100 * torch.matmul(text_features, image_features.T)).softmax(dim=-1)
					similarity = similarity[0]
					
					for similarity_i, value in enumerate(similarity):
						if value > image_confidence:
							image_id_best = self.batch_size * step + similarity_i
							image_confidence = value
						if value > self.candidate_threshold:
							candidates.append([self.batch_size * step + similarity_i, value])
					
					sys.stdout.write(f"STEP {step}, image {step*self.batch_size} / {self.dataset.length} id_best: {image_id_best} filename: {self.dataset.get_filename(image_id_best)} confidence: {image_confidence.item()} candidates: {len(candidates)}\r")
					# TODO Do something with it here
				
				selected_id = image_id_best
				selected_confidence = image_confidence
				# If we found at least 1 item above the confidence threshold, randomly select one to avoid a consistent high scorer from being picked every time
				if candidates:
					picked = random.choice(candidates)
					selected_id = picked[0]
					selected_confidence = picked[1]
				
				obj.media_clip = self.dataset.get_filename(selected_id)
				obj.media_clip_confidence = selected_confidence
				logger.info(f"Tweet {i}: selected {selected_id} → {obj.media_clip} confidence {obj.media_clip_confidence} in {time.time() - time_start}")
				
				handle_out.write(json.dumps(obj) + "\n")
				if i % 100:
					handle_out.flush()
			
