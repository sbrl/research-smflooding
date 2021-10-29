#!/usr/bin/env python3
import io
import sys
import os
from pathlib import Path
import argparse
from loguru import logger
import json

import tensorflow as tf

from lib.io.settings import settings_get, settings_load
from lib.data.CategoryCalculator import CategoryCalculator
from lib.ai.ImageClassifier import ImageClassifier


def init_logging(filepath_output):
	"""Initialises the logging subsystem."""
	
	if filepath_output is not None:
		# Create the output directory if it doesn't exist already
		dirpath = os.path.dirname(filepath_output)
		if not os.path.exists(dirpath):
			os.makedirs(dirpath, 0o750)
		
		# Log to a file - ref https://github.com/conda/conda/issues/9412
		logger.add(io.open(filepath_output, mode="r"))
	
	sys.stderr.write(f"lstm_tweet_classifier: Writing logs to {filepath_output}\n")
	logger.info("lstm_text_classifier init! Here we go")
	logger.info(f"This is Tensorflow {tf.__version__}")


def main():
	"""Main entrypoint."""
	
	parser = argparse.ArgumentParser(description="This program labels tweets using a given model.")
	parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Filepath to either an image to label OR a directory containing images to label.", required=True)
	parser.add_argument("--output", "-o", help="Filepath to write labels to. Labels are written to stdout if not specified.", default="-")
	parser.add_argument("--checkpoint", help="Path to the checkpoint of the model to load.", required=True)
	parser.add_argument("--only-gpu", help="If the GPU is not available, exit with an error (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	
	args = parser.parse_args()
	
	if not os.path.isfile(args.config):
		print("Error: File at '" + args.config + "' does not exist or is not a file.")
		exit(1)
	if not os.path.exists(args.input):
		print("Error: File at '" + args.input + "' does not exist.")
		exit(1)
	if not os.path.isfile(args.checkpoint):
		print("Error: File at '" + args.checkpoint + "' does not exist or is not a file.")
		exit(1)
	
	filepaths_in = args.input
	stream_out = sys.stdout
	
	if args.output:
		stream_out = open(args.output, "w")
	
	settings_load(args.config)
	
	settings = settings_get()
	settings.input = args.input
	settings.checkpoint = args.checkpoint
	init_logging(None)
	
	gpus = tf.config.list_physical_devices('GPU')
	logger.info(f"classify_images: Available gpus: {gpus}")
	tf.__version__
	if not gpus and args.only_gpu:
		logger.info("No GPUs detected, exiting because --only-gpu was specified")
		sys.exit(1)
	
	
	if not settings.data.paths.categories:
		print("Error: No path to the categories file specified (data.paths.categories)")
		sys.exit(1)
	if not os.path.exists(settings.data.paths.categories):
		print(f"Error: No such file or directory {settings.data.paths.categories}")
		sys.exit(2)
	
	if os.path.isdir(filepaths_in):
		filepaths_in = list(map(
			lambda filename : os.path.join(filepaths_in, filename),
			os.listdir(filepaths_in)
		))
	else:
		filepaths_in = [ filepaths_in ]
	
	
	###############################################################################
	
	container = {}
	
	
	
	cats = CategoryCalculator(settings.data.paths.categories)
	
	
	model = ImageClassifier(
		container,
		cats.count,
		settings,
		args.checkpoint
	)
	
	results = model.predict_class_ids(filepaths_in)
	
	for result in results:
		if result is None:
			continue
		
		cat_name = cats.index2name(result[1])
		stream_out.write(result[0]+"\t"+str(cat_name)+"\n")
	
	stream_out.close()


if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
