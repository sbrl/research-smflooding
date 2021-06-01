#!/usr/bin/env python3
import io
import sys
import os
from pathlib import Path
import argparse
import logging
import json
# import tensorflow as tf

from lib.io.settings import settings_get, settings_load
from lib.data.TweetsData import TweetsData
from lib.ai.LSTMTweetClassifier import LSTMTweetClassifier


def init_logging(filepath_output):
	"""Initialises the logging subsystem."""
	
	# Log to a file - ref https://github.com/conda/conda/issues/9412
	logging.basicConfig(level=logging.INFO, filename=filepath_output)
	
	logging.info("lstm_text_classifier init! Here we go")
	sys.stderr.write(f"Writing logs to {filepath_output}")
	
	handle = io.open("/dev/tty", "w")
	handle.write(f"lstm_tweet_classifier: Writing logs to {filepath_output}")
	handle.close()
	sys.stderr.write(f"lstm_tweet_classifier: Writing logs to {filepath_output}")


def main():
	"""Main entrypoint."""
	
	parser = argparse.ArgumentParser(description="This program calculates trains a tweet classification model.")
	parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
	parser.add_argument("--output", "-o", help="Path to output directory to write output to (will be automatically created if it doesn't exist)", required=True)
	
	args = parser.parse_args()
	
	if not Path(args.config).is_file():
		print("Error: File at '" + args.config + "' does not exist.")
		exit(1)
	
	settings_load(args.config)
	
	settings = settings_get()
	settings.output = args.output
	init_logging(os.path.join(
		settings.output,
		"lstm_text_classifier.log"
	))
	
	if not settings.data.paths.glove:
		print("Error: No path to the pre-trained glove txt file specified (data.paths.glove)")
		exit(1)
	if not settings.data.paths.categories:
		print("Error: No path to the categories file specified (data.paths.categories)")
		exit(1)
	if not settings.data.paths.input_train:
		print("Error: No path to the input tweets jsonl file specified to train on (data.paths.input_train)")
		exit(1)
	if not settings.data.paths.input_validate:
		print("Error: No path to the input tweets jsonl file specified to validate with (data.paths.input_validate)")
		exit(1)
	
	###############################################################################
	
	container = {}
	
	dataset_train		= TweetsData(settings.data.paths.input_train, container)
	dataset_validate	= TweetsData(settings.data.paths.input_validate, container)
	
	ai = LSTMTweetClassifier(container)
	ai.train(dataset_train, dataset_validate)
	

if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
