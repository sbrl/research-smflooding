import torch

# from PIL import Image

class CLIPModel(torch.nn.Module):
	def __init__(self, clip_model, units=512, classes=2, device="cpu", **kwargs):
		super(CLIPModel, self).__init__()
		
		self.device = device
		self.clip = clip_model
		self.classifier = torch.nn.Sequential(
			torch.nn.Linear(units*2, units),
			torch.nn.ReLU(),
			torch.nn.Linear(units, units),
			torch.nn.ReLU(),
			torch.nn.Linear(units, classes),
			torch.nn.Softmax(dim=1)
		).to(self.device)
	
	def forward(self, images, text):
		with torch.no_grad():
			text_encoded = self.clip.encode_text(text.int().to(self.device))
			images_encoded = self.clip.encode_image(images.to(self.device))
		
		# Q: >> We *may* need to run this through the model itself, but whether we need to do this or not is currently unclear.
		# 2024-06-13 A: No, we don't need to run it through the 'model itself'. CLIP works by projecting images and text into the same shared latent space. This is simply taking advantage of that for classification purposes.
		
		# text_encoded and images_encoded should look like [ batch_size, 512 ]
		result = torch.stack([text_encoded, images_encoded], -1)
		
		# result = torch.adaptive_avg_pool1d(result, 1).squeeze(-1)
		result = result.flatten(start_dim=1).type(torch.FloatTensor).to(self.device) # [ batch_size, 1024 ]
		result = self.classifier(result) # [ batch_size, 2 ]
		return result
