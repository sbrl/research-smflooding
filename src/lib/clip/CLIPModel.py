import torch

import clip

from PIL import Image

class CLIPModel(torch.nn.Module):
	def __init__(self, units=512, classes=2, device="cpu", **kwargs):
		super(CLIPModel, self).__init__()
		
		self.clip, _ = clip.load("ViT-B/32", device=device)
		self.clip_units_out = 512
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(512, units),
			torch.nn.ReLU(),
			torch.nn.Linear(units, units),
			torch.nn.ReLU(),
			torch.nn.Linear(units, classes),
			torch.nn.Softmax(dim=1)
		).to(device)
	
	def forward(self, text, images):
		with torch.no_grad():
			text_encoded = self.clip.encode_text(text.int())
			images_encoded = self.clip.encode_image(images)
		
		print(f"DEBUG text_encoded shape {text_encoded.shape}")
		print(f"DEBUG images_encoded shape {images_encoded.shape}")
		
		# NOTE: We *may* need to run this through the model itself, but whether we need to do this or not is curently unclear.
		
		# text_encoded and images_encoded should look like [ 1, 512 ]
		result = torch.stack([text_encoded, images_encoded], -1)
		
		print(f"DEBUG result shape after stack {result.shape}")
		result = torch.adaptive_avg_pool1d(result, 1).squeeze(-1)
		print(f"DEBUG result shape {result.shape}")
		return self.classifier(result)
