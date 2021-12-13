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
from lib.data.TweetsImageData import TweetsImageData
from lib.data.CategoryCalculator import CategoryCalculator
from lib.ai.ImageClassifier import ImageClassifier
from lib.ai.ConfusionMatrixMaker import ConfusionMatrixMaker


def init_logging():
	"""Initialises the logging subsystem."""
	
	logger.info("confusion_matrix_image init! Here we go")
	logger.info(f"This is Tensorflow {tf.__version__}")


def parse_args():
	"""Defines and parses the CLI arguments."""
	parser = argparse.ArgumentParser(description="This program creates a confusion matrix for a pre-trained image classification model.")
	parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input-images", "-i", help="Path to input directory that contains the images to use to analyse the AI model")
	parser.add_argument("--input-tweets", "-t", help="Path to input file that contains the LABELLED tweets to use to analyse the AI model", required=True)
	parser.add_argument("--model", "-m", help="Filepath to the AI model checkpoint to load.", required=True)
	parser.add_argument("--output", "-o", help="Path to the output file to write the output as a PNG to.", required=True)
	parser.add_argument("--min-confidence", help="Minimum confidence interval from the AI model to require to include a given prediction in the resulting confusion matrix.")
	parser.add_argument("--only-gpu",
		help="If the GPU is not available, exit with an error (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	
	return parser.parse_args()


def main():
	"""Main entrypoint."""
	init_logging()
	
	args = parse_args()
	
	gpus = tf.config.list_physical_devices('GPU')
	logger.info(f"confusion_matrix_image: Available gpus: {gpus}")
	
	if not gpus and args.only_gpu:
		logger.info("No GPUs detected, exiting because --only-gpu was specified")
		sys.exit(1)
	if not hasattr(args, "min_confidence"):
		args.min_confidence = 0.8
	
	settings_load(
		filepath_custom=args.config
		filename_default="settings.media.default.toml" # Load the media settings 'cause we're working with images
	)
	settings = settings_get()
	
	if not os.path.exists(args.input_tweets) or not os.path.isfile(args.input_tweets):
		print(f"Error: No such file {args.input_tweets} (does it exist, do you have permission to read it, and is it actually a file and not a directory?)")
		sys.exit(2)
	if type(args.input_images) is str and os.path.exists(args.input_images) and not os.path.isdir(args.input_images):
		print(f"Error: No such directory {args.input_images} (does it exist, do you have permission to read it, and is it actually a directory and not a file?)")
		sys.exit(2)
	if not os.path.exists(args.model):
		print(f"Error: No such file or directory {args.model}")
		sys.exit(2)
	if not settings.data.paths.categories:
		print("Error: No categories file specified in the setting data.paths.categories.")
		sys.exit(1)
	if not os.path.exists(settings.data.paths.categories):
		print(f"Error: No such category file '{settings.data.paths.categories}'")
		sys.exit(2)
	
	if args.input_images:
		settings.data.paths.input_media_dir = args.input_images
	
	###############################################################################
	
	container = {}
	
	
	cats = CategoryCalculator(settings.data.paths.categories)
	
	TweetsImageData.init_globals(settings.data.paths.categories)
	dataset_predict = TweetsImageData.generator(args.input_tweets)
	matrix_maker = ConfusionMatrixMaker(
		ImageClassifier(
			container,
			cats.count,
			settings,
			filepath_checkpoint=args.model
		),
		cats,
		min_confidence=args.min_confidence
	)
	matrix_maker.render(
		dataset_predict,
		args.output
	)


if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
