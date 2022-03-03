import torch

import clip
from PIL import Image

class CLIPModel(torch.nn.Module):
	def __init__(self, units=512, classes=2, device="cpu", **kwargs):
		super(CLIPModel, self).__init__()
		
		self.clip, _ = clip.load("ViT-B/32", device=device)
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(units*2, units),
			torch.nn.ReLU(),
			torch.nn.Linear(units, units),
			torch.nn.ReLU(),
			torch.nn.Linear(units, classes),
			torch.nn.Softmax(dim=1)
		).to(device)
	
	def forward(self, images, text):
		with torch.no_grad():
			text_encoded = self.clip.encode_text(text.int())
			images_encoded = self.clip.encode_image(images)
		
		# NOTE: We *may* need to run this through the model itself, but whether we need to do this or not is curently unclear.
		
		# text_encoded and images_encoded should look like [ batch_size, 512 ]
		result = torch.stack([text_encoded, images_encoded], -1)
		
		# result = torch.adaptive_avg_pool1d(result, 1).squeeze(-1)
		result = result.flatten(start_dim=1) # [ batch_size, 1024 ]
		result = self.classifier(result) # [ batch_size, 2 ]
		return result
