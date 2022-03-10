#!/usr/bin/env python3
import io
import sys
import os
from pathlib import Path
import argparse
from loguru import logger
import json

import torch
import clip

from lib.io.settings import settings_get, settings_load

from lib.data.CategoryCalculator import CategoryCalculator
from lib.clip.CLIPDataset import CLIPDataset
from lib.clip.CLIPClassifier import CLIPClassifier



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
	
	device = "cuda" if torch.cuda.is_available() else "cpu"
	
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
	if not os.path.exists(settings.data.paths.categories):
		print(f"Error: No such file or directory {settings.data.paths.categories}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.dir_media):
		print(f"Error: No such file or directory {settings.data.paths.dir_media}")
		sys.exit(2)
	
	
	###############################################################################
	
	###
	## 1: Create datasets
	###
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device).to(self.device)
	cats = CategoryCalculator(settings.data.paths.categories)
	
	dataset_settings_common = {
		"dir_media": settings.data.paths.dir_media,
		"cats": cats,
		"device": settings.model.device,
		"clip_preprocess": clip_preprocess
	}
	
	dataset_train = CLIPDataset(
		filepath_tweets=settings.data.paths.input_train,
		**dataset_settings_common
	)
	dataset_validate = CLIPDataset(
		filepath_tweets=settings.data.paths.input_validate,
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
		device=device
	)
	
	ai.train(dataset_train, dataset_validate)

if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
