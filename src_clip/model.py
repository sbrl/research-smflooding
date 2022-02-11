import torch

import clip

from PIL import Image

class CLIP(torch.nn.Module):
    def _init__(self, units=512, classes=2, device="cpu", **kwargs):
        super(CLIP, self).__init__()
        self.clip = clip.load("ViT-B/32", device=device)
        self.clip_units_out = 512
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, units),
            torch.nn.ReLU(),
            torch.nn.Linear(units, units),
            torch.nn.ReLU(),
            torch.nn.Linear(units, classes),
            torch.nn.Softmax()
        ).to(device)
    
    def forward(self, text, images):
        with torch.no_grad():
            text_encoded = self.clip.encode_text(text)
            images_encoded = self.clip.encode_image(images)
        
        # text_encoded and image_encoded should look like [ 1, 512 ]
        result = torch.stack([text_encoded, image_encoded], 1)
        
        result = torch.adaptive_avg_pool1d(result, 1)
            .squeeze(-1)
        
        return self.classifier(result)
