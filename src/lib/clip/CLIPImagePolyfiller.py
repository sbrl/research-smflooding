import io

from loguru import logger
import torch
import clip

class CLIPImagePolyfiller(object):
	def __init__(self, dataset_images, clip_model, device, batch_size=64):
		super(CLIPImagePolyfiller, self).__init__()
		
		self.device = device
		self.batch_size = batch_size
		
		self.dataset = dataset_images
		self.data = torch.utils.data.Dataloader(
			dataset_images,
			batch_size=self.batch_size,
			shuffle=False
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
			
			text = clip.tokenize(obj["text"].strip(), truncate=True).squeeze(0).to(self.device)
			
			text_batch = torch.stack([text]*self.batch_size, 0)
			
			image_id_best = -1		# The id of the one we're most confident about
			image_confidence = -1	# How confident we are in the current best
			with torch.no_grad():
				for step, image_batch in enumerate(self.data):
					logger.info(f"STEP {step}")
					
					result = self.clip_model(text_batch, image_batch)
					
					print("BATCH_RESULT", result.shape, result)
					
					# TODO Do something with it here
			
			obj.media_clip = "TODO" # self.dataset.get_filename(image_id_best)
			obj.media_clip_confidence = image_confidence
			
			handle_out.write(json.dumps(obj) + "\n")
			if i % 100:
				handle_out.flush()
			
