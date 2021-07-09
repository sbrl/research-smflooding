#!/usr/bin/env python3
import io
import sys
import os
from pathlib import Path
import argparse
import logging
import json

import tensorflow as tf

from lib.io.settings import settings_get, settings_load
from lib.data.TweetsData import TweetsData
from lib.data.CategoryCalculator import CategoryCalculator
from lib.ai.TweetClassifier import TweetClassifier
from lib.ai.ConfusionMatrixMaker import ConfusionMatrixMaker


def init_logging(filepath_output):
	"""Initialises the logging subsystem."""
	
	if filepath_output is None:
		logging.basicConfig(level=logging.INFO)
	else:
		# Create the output directory if it doesn't exist already
		dirpath = os.path.dirname(filepath_output)
		if not os.path.exists(dirpath):
			os.makedirs(dirpath, 0o750)
		
		# Log to a file - ref https://github.com/conda/conda/issues/9412
		logging.basicConfig(level=logging.INFO, filename=filepath_output)
	
	sys.stderr.write(f"confusion_matrix: Writing logs to {filepath_output}\n")
	logging.info("confusion_matrix init! Here we go")
	logging.info(f"This is Tensorflow {tf.__version__}")


def parse_args():
	"""Defines and parses the CLI arguments."""
	parser = argparse.ArgumentParser(description="This program creates a confusion matrix for a pre-trained tweet classification model.")
	parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--input", "-i", help="Path to input file that contains the tweets to use to analyse the AI model", required=True)
	parser.add_argument("--model", "-m", help="Filepath to the AI model checkpoint to load.", required=True)
	parser.add_argument("--output", "-o", help="Path to the output file to write the output as a PNG to.", required=True)
	parser.add_argument("--only-gpu",
		help="If the GPU is not available, exit with an error  (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	
	return parser.parse_args()


def main():
	"""Main entrypoint."""
	
	args = parse_args()
	
	gpus = tf.config.list_physical_devices('GPU')
	logging.info(f"lstm_text_classifier: Available gpus: {gpus}")
	tf.__version__
	if not gpus and args.only_gpu:
		logging.info("No GPUs detected, exiting because --only-gpu was specified")
		sys.exit(1)
	
	
	settings_load(args.config)
	settings = settings_get()
	
	if not os.path.exists(args.input):
		print(f"Error: No such file or directory {args.input}")
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
	
	TweetsData.init_globals(settings.data.paths.categories, settings.data.paths.glove)
	dataset_predict = TweetsData.generator(args.input)
	matrix_maker = ConfusionMatrixMaker(
		TweetClassifier(container, filepath_checkpoint=args.model),
		CategoryCalculator(settings.data.paths.categories)
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