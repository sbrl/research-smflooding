#!/usr/bin/env python3
import io
import sys
import os
from pathlib import Path
import argparse
from loguru import logger
import json

import torch

from lib.clip.CLIPLabeller import CLIPLabeller
from lib.data.CategoryCalculator import CategoryCalculator


def init_logging(filepath_output):
	"""Initialises the logging subsystem."""
	
	if filepath_output is not None:
		# Create the output directory if it doesn't exist already
		dirpath = os.path.dirname(filepath_output)
		if not os.path.exists(dirpath):
			os.makedirs(dirpath, 0o750)
		
		# Log to a file - ref https://github.com/conda/conda/issues/9412
		logger.add(io.open(filepath_output, mode="r"))
	
	sys.stderr.write(f"clip_labeller: Writing logs to {filepath_output}\n")
	logger.info("clip_labeller init! Here we go")
	logger.info(f"This is PyTorch {torch.__version__}")


def main():
	"""Main entrypoint."""
	
	parser = argparse.ArgumentParser(description="This program labels tweets using a previously saved CLIP model.")
	parser.add_argument("--input", "-i", help="Filepath to the file containing the associated tweets. If not specified, data is read from stdin.")
	parser.add_argument("--output", "-o", help="Filepath to write labelled tweets to. If not specified, data is written to stdout.")
	parser.add_argument("--media", "-m", help="Path to the directory containing the associated media items.", required=True)
	parser.add_argument("--checkpoint", help="Path to the checkpoint of the model to load.", required=True)
	parser.add_argument("--device", help="The device to use for compute. Defaults to auto, which uses the GPU if available, but otherwise falls back on the CPU.")
	parser.add_argument("--only-gpu", help="If the GPU is not available, exit with an error (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	
	args = parser.parse_args()
	
	if args.input and not os.path.isfile(args.input):
		sys.stderr.write("Error: File at '" + args.input + "' does not exist.\n")
		exit(1)
	if not os.path.isfile(args.checkpoint):
		sys.stderr.write("Error: File at '" + args.checkpoint + "' does not exist.\n")
		exit(1)
	
	stream_in = sys.stdin
	stream_out = sys.stdout
	dir_media = args.media
	filepath_checkpoint = args.checkpoint
	filepath_cats = args.emoji_cats
	device = "auto"
	only_gpu = False
	
	if args.input:
		stream_in = io.open(args.input, "r")
	if args.output:
		stream_out = io.open(args.output, "w")
	if args.device:
		device = args.device
	if args.only_gpu:
		only_gpu = True
	
	init_logging(None)
	
	
	if device == "auto":
		device = "cuda" if torch.cuda.is_available() else "cpu"
	
	logger.info(f"Compute device selected: {device}")
	
	gpus = torch.cuda.device_count()
	logger.info(f"clip_classifier: {gpus} gpus available")
	if only_gpu and not torch.cuda.is_available():
		logger.info("No GPUs detected, exiting because --only-gpu was specified")
		sys.exit(1)
	
	
	if not os.path.exists(filepath_cats):
		sys.stderr.write(f"Error: No such file or directory {filepath_cats}")
		sys.exit(2)
	if not os.path.exists(filepath_checkpoint):
		sys.stderr.write(f"Error: No such file or directory {filepath_checkpoint}")
		sys.exit(2)
	if not os.path.exists(dir_media):
		sys.stderr.write(f"Error: No such directory {dir_media}\n")
		sys.exit(2)
	
	
	###############################################################################
	
	cats = CategoryCalculator(filepath_cats)
	
	labeller = CLIPLabeller(
		cats=cats,
		filepath_checkpoint=filepath_checkpoint,
		dir_media=dir_media,
		device=device
	)
	
	labeller.label(
		stream_in,
		stream_out
	)
	
	stream_out.close()
	
	logger.info("Labelling complete!")


if __name__ == "__main__":
	main()
else:
	sys.stderr.write("This script must be run directly. It cannot be imported.\n")
	exit(1)
