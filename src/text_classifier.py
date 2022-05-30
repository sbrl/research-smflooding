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
from lib.data.TweetsData import TweetsData
from lib.ai.TweetClassifier import TweetClassifier


def init_logging(filepath_output):
	"""Initialises the logging subsystem."""
	
	if filepath_output is not None:
		# Create the output directory if it doesn't exist already
		dirpath = os.path.dirname(filepath_output)
		if not os.path.exists(dirpath):
			os.makedirs(dirpath, 0o750)
		
		# Log to a file - ref https://github.com/conda/conda/issues/9412
		logger.add(io.open(filepath_output, mode="w"))
	
	sys.stderr.write(f"text_classifier: Writing logs to {filepath_output}\n")
	logger.info("text_classifier init! Here we go")
	logger.info(f"This is Tensorflow {tf.__version__}")


def parse_args():
	"""Defines and parses the CLI arguments."""
	parser = argparse.ArgumentParser(description="This program trains an emoji-based tweet classification model.")
	parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--output", "-o", help="Path to output directory to write output to (will be automatically created if it doesn't exist)", required=True)
	parser.add_argument("--log-stdout", help="Log to stdout, rather than a log file", action="store_true")
	parser.add_argument("--only-gpu",
		help="If the GPU is not available, exit with an error  (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	parser.add_argument("--nobidi", help="Don't add the Bidirectional wrapper to the LSTM layers", action="store_true")
	# parser.add_argument("--smoteify", help="Apply SMOTE to the input training data. The training data should be UNBALANCED for this to work!", action="store_true")
	parser.add_argument("--batchnorm", help="Enable Batch Normalisation", action="store_true")
	parser.add_argument("--model", help="The type of model to create [training only; default: lstm].", choices=["lstm", "transformer"])
	parser.add_argument("--batch-size", help="Sets the batch size.", type=int)
	parser.add_argument("--no-tensorboard", help="Disables Tensorboard data generation.", action="store_true")
	
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
	if hasattr(args, "no_tensorboard") and args.no_tensorboard:
		settings.train.tensorboard_enable = False
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
	
	gpus = tf.config.list_physical_devices('GPU')
	logger.info(f"text_classifier: Available gpus: {gpus}")
	tf.__version__
	if not gpus and args.only_gpu:
		logger.info("No GPUs detected, exiting because --only-gpu was specified")
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
		settings.data.paths.input_train, container
		# smote=settings.train.smoteify
	)
	dataset_validate	= TweetsData(
		settings.data.paths.input_validate, container
		# smote=False
	)
	
	ai = TweetClassifier(container, do_tensorboard=settings.train.tensorboard_enable)
	ai.train(dataset_train, dataset_validate)
	

if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
