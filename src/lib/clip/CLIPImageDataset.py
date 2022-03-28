import os

from loguru import logger

import torch
import torchvision
# from simplejpeg import decode_jpeg

class CLIPImageDataset(torch.utils.data.Dataset):
	
	def __init__(self, dir_media, device, clip_preprocess):
		super(CLIPImageDataset).__init__()
		
		self.dir_media = dir_media
		self.device = device
		
		self.preprocess = clip_preprocess
		
		self.files = os.listdir(self.dir_media)
		self.files = [ os.path.join(self.dir_media, filename) for filename in self.files if not os.path.isdir(filename) ]
		self.length = len(self.files)
		logger.info(f"Loaded {self.length} image filenames")
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, image_id):
		return self.preprocess(
			torchvision.io.read_image(self.files[image_id])
		).to(device=self.device)
		# return self.preprocess(Image.open(self.files[image_id])).to(device=self.device)
	
	
	# If one knows the batch size and the number of batches processed, one can determine the index of the filename in question
	def get_filename(self, image_id):
		return self.files[image_id]
