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
from lib.data.TweetsDataSimple import tweets_data_simple


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
	parser.add_argument("--output", "-o", help="Path to a directory to write models and their outputs to. If it does not exist it will be created.", required=True)
	parser.add_argument("--topic-count", "-t", help="The number of topics to group the input tweets into [default: 10].", type=int)
	parser.add_argument("--words-per-topic", "-n", help="The number of words per topic to return [default].", type=int)
	parser.add_argument("--checkpoint", help="The filepath to the *directory* that contains the model to load.")
	
	args = parser.parse_args()
	
	if args.input and args.input != "-" and not os.path.isfile(args.input):
		print("Error: File at '" + args.input + "' does not exist.")
		exit(1)
	
	model = "lda"
	stream_in = sys.stdin
	stream_out = sys.stdout
	
	if args.model:
		model = args.model
	if args.input:
		stream_in = open(args.input, "r")
	if args.output:
		stream_output = open(args.output, "r")
	if not args.topic_count: # Seriously, why is it so difficult to specify a default value for a CLI arg?!
		args.topic_count = 10
	if not args.words_per_topic:
		args.words_per_topic = 10
	if not os.path.isdir(args.output):
		os.makedirs(args.output)
	
	init_logging(None)
	
	###########################################################################
	
	filepath_model = os.path.join(args.output, "model.gensim.bin")
	container = {}
	
	
	ai = TopicAnalyser(
		args.topic_count,
		args.words_per_topic
	)
	ai.load(filepath_model)
	
	labeller = TopicLabeller(ai)
	labeller.label(stream_in, stream_out)
	# TODO: Implement TopicLabeller class here


if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
