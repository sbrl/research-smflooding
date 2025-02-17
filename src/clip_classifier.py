#!/usr/bin/env python3
import io
import sys
import os
from pathlib import Path
import argparse
from loguru import logger

import torch
import clip

from lib.io.settings import settings_get, settings_load

from lib.data.CategoryCalculator import CategoryCalculator
from lib.clip.CLIPDataset import CLIPDataset
from lib.clip.CLIPClassifier import CLIPClassifier


"""
This program trains a CLIP-based classifier for multimodal data (images and text).
It provides functionality for training and validating using a CLIP model.
The CLIPClassifier class encapsulates the model, loss function, and optimization process.
It supports checkpointing and logging metrics during training.
"""


def init_logging(filepath_output):
	"""Initialises the logging subsystem."""
	
	if filepath_output is not None:
		# Create the output directory if it doesn't exist already
		dirpath = os.path.dirname(filepath_output)
		if not os.path.exists(dirpath):
			os.makedirs(dirpath, 0o750)
		
		# Log to a file - ref https://github.com/conda/conda/issues/9412
		logger.add(io.open(filepath_output, mode="w"))
	
	sys.stderr.write(f"clip_classifier: Writing logs to {filepath_output}\n")
	logger.info("clip_classifier init! Here we go")
	logger.info(f"This is PyTorch {torch.__version__}")


def parse_args():
	"""Defines and parses the CLI arguments."""
	parser = argparse.ArgumentParser(description="This program trains an emoji-based tweet classification model.")
	parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--output", "-o", help="Path to output directory to write output to (will be automatically created if it doesn't exist)", required=True)
	parser.add_argument("--log-stdout", help="Log to stdout, rather than a log file", action="store_true")
	parser.add_argument("--only-gpu",
		help="If the GPU is not available, exit with an error  (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	parser.add_argument("--batch-size", help="Sets the batch size.", type=int)
	parser.add_argument("--clip-media-threshold", help="If a tweet has a clip-assigned image via data augmentation, any with a confidence below this value will be discarded. Must be between 0 and 1 (default: 0.75).", type=float)
	parser.add_argument("--no-do-images", help="You do not want this option. Set all images to a blank white image instead of loading the actual images from disk. Useful only in ablative studies etc.", action="store_true")
	parser.add_argument("--seed", "-s", help="Set the global random seed. You do not want this option. Useful only for direct comparative studies, but basically useless since the ONNX runtime scrambles the determinism and reproduceablility anyway.", type=int)
	parser.add_argument("--units", "-u", help="The number of units/params/etc to use INSIDE the tail-end model we've tacked onto CLIP. You can freely change this value, because the units in to our little submodel is set separately internally.", type=int)
	
	return parser.parse_args()


def main():
	"""Main entrypoint."""
	
	args = parse_args()
	
	if not Path(args.config).is_file():
		print("Error: File at '" + args.config + "' does not exist.")
		exit(1)
	
	settings_load(
		filepath_custom=args.config,
		filename_default="settings.clip.default.toml"
	)
	
	settings = settings_get()
	if hasattr(args, "batch_size") and type(args.batch_size) is int:
		settings.train.batch_size = args.batch_size
	if hasattr(args, "clip_media_threshold") and type(args.clip_media_threshold) is float:
		settings.data.clip_label_threshold = args.clip_media_threshold
	if hasattr(args, "units") and type(args.units) is int:
		settings.model.units = args.units # Default: 512, set in settings.clip.default.toml
	
	# if hasattr(args, "smoteify") and args.smoteify:
	# 	settings.train.smoteify = True
	settings.output = args.output
	if args.log_stdout:
		init_logging(None)
	else:
		init_logging(os.path.join(
			settings.output,
			"this_run.log"
		))
	
	
	if hasattr(args, "seed") and type(args.seed) is int:
		torch.manual_seed(args.seed)
		logger.warning(f"Randomness seed set to {str(args.seed)}.")
	
	
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if settings.model.device != "auto":
		device = settings.model.device
	
	logger.info(f"Compute device selected: {device}")
	
	gpus = torch.cuda.device_count()
	logger.info(f"clip_classifier: {gpus} gpus available")
	if not torch.cuda.is_available() and args.only_gpu:
		logger.info("No GPUs detected, exiting because --only-gpu was specified")
		sys.exit(1)
	
	if not settings.data.paths.categories or len(settings.data.paths.categories) == 0:
		print("Error: No path to the categories file specified (data.paths.categories)")
		sys.exit(1)
	if not settings.data.paths.input_train or len(settings.data.paths.input_train) == 0:
		print("Error: No path to the input tweets jsonl file specified to train on (data.paths.input_train)")
		sys.exit(1)
	if not settings.data.paths.input_validate or len(settings.data.paths.input_validate) == 0:
		print("Error: No path to the input tweets jsonl file specified to validate with text,(data.paths.input_validate)")
		sys.exit(1)
	if not settings.data.paths.dir_media or len(settings.data.paths.dir_media) == 0:
		print("Error: No path to the media directory specified (data.paths.dir_media)")
	
	if not os.path.exists(settings.data.paths.input_train):
		print(f"Error: No such file or directory {settings.data.paths.input_train}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.input_validate):
		print(f"Error: No such file or directory {settings.data.paths.input_validate}")
		sys.exit(2)
	if settings.data.paths.input_test and len(settings.data.paths.input_test) > 0 and not os.path.exists(settings.data.paths.input_test):
		print(f"Error: data.paths.input_test was specified, but no such file or directory '{settings.data.paths.input_test}' exists.")
		sys.exit(2)
		
	if not os.path.exists(settings.data.paths.categories):
		print(f"Error: No such file or directory {settings.data.paths.categories}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.dir_media):
		print(f"Error: No such file or directory {settings.data.paths.dir_media}")
		sys.exit(2)
	
	if type(settings.data.clip_label_threshold) is not float:
		print("Error: clip_label_threshold is not of type float.")
		sys.exit(3)
	
	###############################################################################
	
	###
	## 1: Create datasets
	###
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
	cats = CategoryCalculator(settings.data.paths.categories)
	
	dataset_settings_common = {
		"dir_media": settings.data.paths.dir_media,
		"cats": cats,
		"device": device,
		"clip_preprocess": clip_preprocess,
		"clip_label_threshold": settings.data.clip_label_threshold,
		"do_images": not args.no_do_images
	}
	
	logger.info(f"DEBUG:dataset_settings_common {str(dataset_settings_common)}")
	
	dataset_train = CLIPDataset(
		filepath_tweets=settings.data.paths.input_train,
		**dataset_settings_common
	)
	dataset_validate = CLIPDataset(
		filepath_tweets=settings.data.paths.input_validate,
		**dataset_settings_common
	)
	dataset_test = None
	if settings.data.paths.input_test and len(settings.data.paths.input_test) > 0:
		dataset_test = CLIPDataset(
			filepath_tweets=settings.data.paths.input_test,
			**dataset_settings_common
		)
	
	
	###
	## 2: Create AI model & train
	###
	ai = CLIPClassifier(
		dir_output=args.output,
		epochs=settings.train.epochs,
		batch_size=settings.train.batch_size,
		clip_model=clip_model,
		device=device,
		units=settings.model.units
	)
	ai.train(dataset_train, dataset_validate, dataset_test)

if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
