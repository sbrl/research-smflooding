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
from lib.data.TweetsImageData import TweetsImageData
from lib.ai.ImageClassifier import ImageClassifier


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
	
	sys.stderr.write(f"tweet_media_classifier: Writing logs to {filepath_output}\n")
	logging.info("tweet_media_classifier init! Here we go")
	logging.info(f"This is Tensorflow {tf.__version__}")


def parse_args():
	"""Defines and parses the CLI arguments."""
	parser = argparse.ArgumentParser(description="This program calculates trains a tweet media image classification model.")
	parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--output", "-o", help="Path to output directory to write output to (will be automatically created if it doesn't exist)", required=True)
	parser.add_argument("--log-stdout", help="Log to stdout, rather than a log file", action="store_true")
	parser.add_argument("--only-gpu",
		help="If the GPU is not available, exit with an error  (useful on shared HPC systems to avoid running out of memory & affecting other users)", action="store_true")
	parser.add_argument("--batch-size", help="Sets the batch size.", type=int)
	parser.add_argument("--image-size", help="Sets the image size. All input images are resized to match this size before continuing through the rest of the model. Aspect ratio is not preserved.", type=int)
	parser.add_argument("--fashion-mnist", help="Use fashion MNIST instead of a regular dataset.", action="store_true")
	
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
	if hasattr(args, "image_size") and type(args.image_size) is int:
		settings.model.image_size = args.image_size
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
	
	
	if not settings.data.paths.input_train:
		print("Error: No path to the input tweets jsonl file specified to train on (data.paths.input_train)")
		sys.exit(1)
	if not settings.data.paths.input_validate:
		print("Error: No path to the input tweets jsonl file specified to validate with (data.paths.input_validate)")
		sys.exit(1)
	if not settings.data.paths.input_validate:
		print("Error: No path to the media directory specified (data.paths.input_media_dir)")
		sys.exit(1)
	
	if not os.path.exists(settings.data.paths.input_train):
		print(f"Error: No such file or directory {settings.data.paths.input_train}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.input_validate):
		print(f"Error: No such file or directory {settings.data.paths.input_validate}")
		sys.exit(2)
	if not os.path.exists(settings.data.paths.input_media_dir):
		print(f"Error: No such file or directory {settings.data.paths.input_media_dir}")
		sys.exit(2)
	
	###############################################################################
	
	container = {}
	
	
	if args.fashion_mnist:
		logging.info("Loading fashion mnist")
		dataset_train, dataset_validate = tf.keras.datasets.fashion_mnist.load_data()
		print("DEBUG dataset_train", dataset_train[0].shape)
		dataset_train = (
			tf.stack([dataset_train[0], dataset_train[0], dataset_train[0]], axis=-1),
			tf.one_hot(dataset_train[1], depth=10, axis=-1)
		)
		dataset_validate = (
			tf.stack([dataset_validate[0], dataset_validate[0], dataset_validate[0]], axis=-1),
			tf.one_hot(dataset_validate[1], depth=10, axis=-1)
		)
		
		settings.model.image_size = 28
		print("DEBUG dataset_train[1]", dataset_train[1])
		
		class_count = 10
	else:
		logging.info("Loading data")
		dataset_train		= TweetsImageData(
			settings.data.paths.input_train, container
		)
		dataset_validate	= TweetsImageData(
			settings.data.paths.input_validate, container
		)
		class_count = CategoryCalculator(settings.data.paths.categories).count
	
	ai = ImageClassifier(container, class_count, settings)
	ai.setup()
	ai.train(dataset_train, dataset_validate, mode="plain")
	

if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
