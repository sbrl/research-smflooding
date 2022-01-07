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

import gensim

from lib.lda.LDAAnalyser import LDAAnalyser
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
	
	sys.stderr.write(f"lda_topics: Writing logs to {filepath_output}\n")
	logger.info("lda_topics init! Here we go")
	logger.info(f"This is Tensorflow {gensim.__version__}")


def main():
	"""Main entrypoint."""
	
	parser = argparse.ArgumentParser(description="This program extracts common topics from tweets using an LDA model.")
	parser.add_argument("--input", "-i", help="Filepath to the file containing the associated tweets. If not specified, data is read from stdin.")
	parser.add_argument("--output", "-o", help="Path to a directory to write models and their outputs to. If it does not exist it will be created.", required=True)
	parser.add_argument("--topic-count", "-t", help="The number of topics to group the input tweets into.", nargs="?", const=10, type=int)
	
	args = parser.parse_args()
	
	if args.input and args.input != "-" and not os.path.isfile(args.input):
		print("Error: File at '" + args.input + "' does not exist.")
		exit(1)
	
	topic_count = args.topic_count
	stream_in = sys.stdin
	
	if args.input:
		stream_in = open(args.input, "r")
	if not os.path.isdir(args.output):
		os.makedirs(args.output)
	
	init_logging(None)
	
	###########################################################################
	
	container = {}
	
	tweets = tweets_data_simple(args.input)
	
	ai = LDAAnalyser(
		args.topic_count
	)
	
	avg_topic_coherence, topics = ai.train(tweets)
	
	###########################################################################
	
	filepath_settings = os.path.join(args.output, "settings.txt")
	filepath_model = os.path.join(args.output, "model.gensim.bin")
	
	ai.save(filepath_model)
	
	handle = io.open(args.filepath_settings, "w")
	handle.write(f"topic_count	{topic_count}\n")
	handle.write(f"filepath_input	{args.input}\n")
	handle.write(f"average topic coherence	{avg_topic_coherence}\n")
	handle.write(f"datetime on finish	{datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f%z')}\n")
	handle.write(f"training system hostname	{socket.gethostname()}\n")
	handle.close()
	
	


if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
