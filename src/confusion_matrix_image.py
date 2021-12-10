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
	parser.add_argument("--input", "-i", help="Path to input directory that contains the images to use to analyse the AI model", required=True)
	parser.add_argument("--model", "-m", help="Filepath to the AI model checkpoint to load.", required=True)
	parser.add_argument("--output", "-o", help="Path to the output file to write the output as a PNG to.", required=True)
	parser.add_argument("--min-confidence", help="Minimum confidence interval from the AI model to require to include a given prediction in the resulting confusion matrix.")
	parser.add_argument("--only-gpu",
		help="If the GPU is not available, exit with an error  (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	
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
	
	settings_load(args.config)
	settings = settings_get()
	
	if not os.path.exists(args.input) or not os.path.isdir(args.input):
		print(f"Error: No such directory {args.input} (do it exist, do you have permission to read it, and is it actually a directory and not a file?)")
		sys.exit(2)
	if not os.path.exists(args.model):
		print(f"Error: No such file or directory {args.model}")
		sys.exit(2)
	if not settings.data.paths.categories:
		print("Error: No categories file specified in the setting data.paths.categories.")
		sys.exit(1)
	if not settings.data.paths.glove:
		print("Error: No glove file specified in the setting data.paths.glove.")
		sys.exit(1)
	if not os.path.exists(settings.data.paths.categories):
		print(f"Error: No such file or directory {settings.data.paths.categories}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.glove):
		print(f"Error: No such file or directory {settings.data.paths.glove}")
		sys.exit(2)
	
	###############################################################################
	
	container = {}
	
	
	
	TweetsImageData.init_globals(settings.data.paths.categories, settings.data.paths.glove)
	dataset_predict = TweetsImageData.generator(args.input)
	matrix_maker = ConfusionMatrixMaker(
		ImageClassifier(container, filepath_checkpoint=args.model),
		CategoryCalculator(settings.data.paths.categories),
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
