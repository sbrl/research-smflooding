#!/usr/bin/env python3
import io
import sys
import os
import socket
from pathlib import Path
import argparse
from loguru import logger
import json
from datetime import datetime
from pprint import pprint

import pysnooper
import gensim

from lib.topic.TopicAnalyser import TopicAnalyser
from lib.topic.TopicLabeller import TopicLabeller


def init_logging(filepath_output):
	"""Initialises the logging subsystem."""
	
	if filepath_output is not None:
		# Create the output directory if it doesn't exist already
		dirpath = os.path.dirname(filepath_output)
		if not os.path.exists(dirpath):
			os.makedirs(dirpath, 0o750)
		
		# Log to a file - ref https://github.com/conda/conda/issues/9412
		logger.add(io.open(filepath_output, mode="r"))
	
	sys.stderr.write(f"label_tweets_topics: Writing logs to {filepath_output}\n")
	logger.info("label_tweets_topics init! Here we go")
	logger.info(f"This is gensim {gensim.__version__}")


def main():
	"""Main entrypoint."""
	
	parser = argparse.ArgumentParser(description="Labels tweets with a given LDA model.")
	parser.add_argument("--input", "-i", help="Filepath to the file containing the associated tweets to label. If not specified, data is read from stdin.")
	parser.add_argument("--output", "-o", help="Path to a directory to write labelled tweets to. Defaults to stdout.")
	parser.add_argument("--checkpoint", help="The filepath to the *directory* that contains the model to load.", required=True)
	
	args = parser.parse_args()
	
	if args.input and args.input != "-" and not os.path.isfile(args.input):
		print("Error: File at '" + args.input + "' does not exist.")
		exit(1)
	
	stream_in = sys.stdin
	stream_out = sys.stdout
	
	if args.input:
		stream_in = open(args.input, "r")
	if args.output:
		stream_out = open(args.output, "w")
	
	init_logging(None)
	
	###########################################################################
	
	filepath_model = os.path.join(args.checkpoint, "model.gensim.bin")
	container = {}
	
	
	ai = TopicAnalyser()
	ai.load(filepath_model)
	
	labeller = TopicLabeller(ai)
	labeller.label(stream_in, stream_out)
	# TODO: Implement TopicLabeller class here


if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
