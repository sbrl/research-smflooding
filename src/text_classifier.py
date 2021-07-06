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
from lib.ai.TweetClassifier import TweetClassifier


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
	
	sys.stderr.write(f"lstm_tweet_classifier: Writing logs to {filepath_output}\n")
	logging.info("lstm_text_classifier init! Here we go")
	logging.info(f"This is Tensorflow {tf.__version__}")


def parse_args():
	"""Defines and parses the CLI arguments."""
	parser = argparse.ArgumentParser(description="This program calculates trains a tweet classification model.")
	parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--output", "-o", help="Path to output directory to write output to (will be automatically created if it doesn't exist)", required=True)
	parser.add_argument("--log-stdout", help="Log to stdout, rather than a log file", action="store_true")
	parser.add_argument("--only-gpu",
		help="If the GPU is not available, exit with an error  (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	parser.add_argument("--nobidi", help="Don't add the Bidirectional wrapper to the LSTM layers", action="store_true")
	parser.add_argument("--batchnorm", help="Enable Batch Normalisation", action="store_true")
	parser.add_argument("--model", help="The type of model to create [training only; default: lstm].", choices=["lstm", "transftf.__version__ormer"])
	parser.add_argument("--batch-size", help="Sets the batch size.", type=int)
	
	return parser.parse_args()


def main():
	"""Main entrypoint."""
	
	args = parse_args()
	
	if not Path(args.config).is_file():
		print("Error: File at '" + args.config + "' does not exist.")
		exit(1)
	
	settings_load(args.config)
	
	settings = settings_get()
	if hasattr(args, "nobidi") and args.nobidi:
		settings.model.bidirectional = False
	if hasattr(args, "batchnorm") and args.batchnorm:
		settings.model.batch_normalisation = True
	if hasattr(args, "model") and args.model is not None:
		settings.model.type = args.model
	if hasattr(args, "batch_size") and type(args.batch_size) is int:
		settings.train.batch_size = args.batch_size
	settings.output = args.output
	if args.log_stdout:
		init_logging(None)
	else:
		init_logging(os.path.join(
			settings.output,
			"lstm_text_classifier.log"
		))
	
	gpus = tf.config.list_physical_devices('GPU')
	logging.info(f"lstm_text_classifier: Available gpus: {gpus}")
	tf.__version__
	if not gpus and args.only_gpu:
		logging.info("No GPUs detected, exiting because --only-gpu was specified")
		sys.exit(1)
	
	
	if not settings.data.paths.glove:
		print("Error: No path to the pre-trained glove txt file specified (data.paths.glove)")
		sys.exit(1)
	if not settings.data.paths.categories:
		print("Error: No path to the categories file specified (data.paths.categories)")
		sys.exit(1)
	if not settings.data.paths.input_train:
		print("Error: No path to the input tweets jsonl file specified to train on (data.paths.input_train)")
		sys.exit(1)
	if not settings.data.paths.input_validate:
		print("Error: No path to the input tweets jsonl file specified to validate with (data.paths.input_validate)")
		sys.exit(1)
	
	if not os.path.exists(settings.data.paths.input_train):
		print(f"Error: No such file or directory {settings.data.paths.input_train}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.input_validate):
		print(f"Error: No such file or directory {settings.data.paths.input_validate}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.categories):
		print(f"Error: No such file or directory {settings.data.paths.categories}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.glove):
		print(f"Error: No such file or directory {settings.data.paths.glove}")
		sys.exit(2)
	
	
	###############################################################################
	
	container = {}
	
	dataset_train		= TweetsData(
		settings.data.paths.input_train, container,
		"train"
	)
	dataset_validate	= TweetsData(
		settings.data.paths.input_validate, container,
		"validate"
	)
	
	ai = TweetClassifier(container)
	ai.train(dataset_train, dataset_validate)
	

if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
