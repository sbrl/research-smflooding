import os
import io

from loguru import logger

import torch
import torchvision
import simplejpeg

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

class CLIPImageDataset(torch.utils.data.Dataset):
	
	def __init__(self, dir_media, device, image_size): # clip_preprocess
		super(CLIPImageDataset).__init__()
		logger.info(f"IMAGE SIZE {image_size}")
		self.dir_media = dir_media
		self.device = device
		
		# self.preprocess = clip_preprocess
		self.image_size = image_size
		# From https://github.com/openai/CLIP/blob/40f5484c1c74edd83cb9cf687c6ab92b28d8b656/clip/clip.py#L78 with the conversion to RGB removed so we can use simplejpeg instead
		self.preprocess = Compose([
			Resize(image_size, interpolation=InterpolationMode.BICUBIC),
			CenterCrop(image_size),
			# ToTensor(), # Not needed, as we're feeding tensors in rather than PIL images
			Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
		])
		
		self.files = os.listdir(self.dir_media)
		self.files = [ os.path.join(self.dir_media, filename) for filename in self.files if not os.path.isdir(filename) ]
		self.length = len(self.files)
		logger.info(f"Loaded {self.length} image filenames")
	
	def __len__(self):
		return self.length
	
	def __getitem__(self, image_id):
		filename = self.files[image_id].replace(".png", ".jpg")
		# Note: You need to run this command to convert everything to jpeg first:
		# find path/to/media -iname '*.png' | xargs --verbose -n3 -P "$(nproc)" mogrify -format jpg
		# TODO: Replace png → jpg
		image = None
		try:
			with io.open(filename, "rb") as handle:
				image = simplejpeg.decode_jpeg(handle.read(), fastdct=True, fastupsample=True)
		except Exception as error:
			logger.warning(f"Caught error: {error}")
			return None
		
		image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1)
		
		return self.preprocess(image).to(device=self.device)
		# return self.preprocess(Image.open(self.files[image_id])).to(device=self.device)
	
	
	# If one knows the batch size and the number of batches processed, one can determine the index of the filename in question
	def get_filename(self, image_id):
		return self.files[image_id]
