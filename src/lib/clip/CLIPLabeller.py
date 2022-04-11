import io
import sys
import json
import random
import os

from loguru import logger

import torch
import clip
from PIL import Image

from .CLIPClassifier import CLIPClassifier

class CLIPLabeller(object):
    """Labels tweets with a CLIP model."""
    def __init__(self, cats, filepath_checkpoint, dir_media, device, batch_size=64):
        super(CLIPLabeller, self).__init__()
        
        self.device = device
        self.dir_media = dir_media
        self.batch_size = batch_size
        
        self.cats = cats
        self.preprocess, self.clip_model = clip.load("ViT-B/32", device=self.device)
        
        self.ai = CLIPClassifier(clip_model=self.clip_model)
        # HACK: Loading the optimiser currently crashes with "ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group"
        self.ai.checkpoint_load(filepath_checkpoint, load_optimiser=False)
    
    
    def normalise_media_filename(self, url):
        """Normalises a media url to it's associated filepath."""
        return os.path.join(
            self.dir_media,
            os.path.basename(media)
        )
    
    def extract_media_filepath(self, obj):
        """Extracts the filepath to an associated media object, or None if no suitable media item was found."""
        if "media" in obj:
            media_items = list(filter(
                lambda media_item: media_item["type"] == "photo" and os.path.exists(self.normalise_media_filename(media_item["url"])),
                obj["media"]
            ))
            if media_items: # ... is not empty
                # Pick a random media item to pair with the text
                return self.normalise_media_filename(random.choice(media_items)["url"])
        
        if "media_clip" in obj and obj["media_clip"] is not None:
            return self.normalise_media_filename(obj["media_clip"])
        
        return None
    
    def label_batch(self, objs, filepaths_media):
        """Labels a single batch of objects."""
        text = [ clip.tokenize(obj["text"], truncate=True).squeeze(0).to(self.device) for obj in objs ]
        images = [ self.preprocess(Image.open(filepath)).to(self.device) for filepath in filepaths_media ]
        
        predictions = self.ai.predict(images, text)
        
        for i, prediction_index in enumerate(predictions):
            objs[i]["label"] = self.cats.index2name(prediction_index.item())
    
    def label(self, handle_in, handle_out):
        """Reads all JSON objects from filepath_in, labels them with the currently loaded model, and writes them to filepath_out."""
        
        acc = []
        acc_media = []
        i = 0
        for line in handle_in:
            obj = None
            try:
                obj = json.loads(line)
            except Exception as error:
                logger.warning(f"Encountered error parsing line {i+1}: {error}, skipping")
                continue
            
            filepath_media = self.extract_media_filepath(obj)
            
            if filepath_media is None:
                handle_out.write(json.dumps(obj) + "\n")
                continue
            
            acc.append(obj)
            acc.append(filepath_media)
            
            if len(acc) >= self.batch_size:
                label_batch(acc, acc_media)
                for obj in acc:
                    handle_out.write(json.dumps(obj) + "\n")
                acc.clear()
            
            if i % 250:
                sys.stdout.write(f"Labelled tweet {i}")
            
            i += 1
    
