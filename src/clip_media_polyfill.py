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

from lib.clip.CLIPImageDataset import CLIPImageDataset
from lib.clip.CLIPImagePolyfiller import CLIPImagePolyfiller



def init_logging(filepath_output):
	"""Initialises the logging subsystem."""
	
	if filepath_output is not None:
		# Create the output directory if it doesn't exist already
		dirpath = os.path.dirname(filepath_output)
		if not os.path.exists(dirpath):
			os.makedirs(dirpath, 0o750)
		
		# Log to a file - ref https://github.com/conda/conda/issues/9412
		logger.add(io.open(filepath_output, mode="w"))
	
	sys.stderr.write(f"clip_media_polyfill: Writing logs to {filepath_output}\n")
	logger.info("clip_media_polyfill init! Here we go")
	logger.info(f"This is PyTorch {torch.__version__}")


def parse_args():
	"""Defines and parses the CLI arguments."""
	parser = argparse.ArgumentParser(description="This program annotates tweets as to which images best fit them.")
	parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Path to the file to read tweets from.", required=True)
	parser.add_argument("--output", "-o", help="Path to the file to write tweets to.", required=True)
	parser.add_argument("--media", "-m", help="Directory containing media. Overrides the value defined in the config file.", required=True)
	parser.add_argument("--only-gpu",
		help="If the GPU is not available, exit with an error (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
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
	if hasattr(args, "media") and type(args.media) is str:
		settings.data.paths.dir_media = args.media
	settings.input = args.input
	settings.output = args.output
	
	init_logging(None)
	
	device = "cuda" if torch.cuda.is_available() else "cpu"
	if settings.model.device != "auto":
		device = settings.model.device
	
	logger.info(f"Compute device selected: {device}")
	
	gpus = torch.cuda.device_count()
	logger.info(f"clip_media_polyfill: {gpus} gpus available")
	if not torch.cuda.is_available() and args.only_gpu:
		logger.info("No GPUs detected, exiting because --only-gpu was specified")
		sys.exit(1)
	
	
	if not settings.input or len(settings.input) == 0:
		print(f"Error: No path to the input file specified.")
		sys.exit(2)
	if not settings.output or len(settings.output) == 0:
		print(f"Error: No path to the output file specified.")
		sys.exit(2)
	if not settings.data.paths.categories or len(settings.data.paths.categories) == 0:
		print("Error: No path to the categories file specified (data.paths.categories)")
		sys.exit(1)
	if not settings.data.paths.dir_media or len(settings.data.paths.dir_media) == 0:
		print("Error: No path to the media directory specified (data.paths.dir_media)")
	
	if not os.path.exists(settings.input):
		print(f"Error: No such file or directory {settings.input}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.categories):
		print(f"Error: No such file or directory {settings.data.paths.categories}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.dir_media):
		print(f"Error: No such file or directory {settings.data.paths.dir_media}")
		sys.exit(2)
	
	
	###############################################################################
	
	###############################################################################
	# TODO: EVERYTHING BELOW HERE NEEDS CHANGING
	###############################################################################
	
	###
	## 1: Create datasets
	###
	clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
	
	dataset_images = CLIPImageDataset(
		dir_media=settings.data.paths.dir_media,
		device=device,
		clip_preprocess=clip_preprocess
	)
	
	###
	## 2: Create AI model & train
	###
	polyfiller = CLIPImagePolyfiller(
		dataset_images=dataset_images,
		clip_model=clip_model,
		device=device,
		batch_size=settings.train.batch_size
	)
	
	polyfiller.label(settings.input, settings.output)

if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
