#!/usr/bin/env python3
import io
import sys
import os
from pathlib import Path
import argparse
from loguru import logger
import json

from lib.io.settings import settings_get, settings_load
from lib.io.TweetCatsLabeller import TweetCatsLabeller
from lib.data.CategoryCalculator import CategoryCalculator

"""
This file contains the main script for labeling tweets using predefined emoji categories.
It uses the TweetCatsLabeller class from src/lib/io/TweetCatsLabeller.py to process
input tweets, assign labels based on categories, and output the labeled tweets.
The script handles command-line arguments, file I/O, and logging.
"""


def init_logging(filepath_output):
	"""Initialises the logging subsystem."""
	
	if filepath_output is not None:
		# Create the output directory if it doesn't exist already
		dirpath = os.path.dirname(filepath_output)
		if not os.path.exists(dirpath):
			os.makedirs(dirpath, 0o750)
		
		# Log to a file - ref https://github.com/conda/conda/issues/9412
		logger.add(io.open(filepath_output, mode="r"))
	
	sys.stderr.write(f"label_tweets_cats: Writing logs to {filepath_output}\n")
	logger.info("label_tweets_cats init! Here we go")


def main():
	"""Main entrypoint."""
	
	parser = argparse.ArgumentParser(description="This program labels tweets using a given categories file.")
	parser.add_argument("--cats", "-c", help="Filepath to the categories file to load.", required=True)
	parser.add_argument("--input", "-i", help="Filepath to the file containing the associated tweets. If not specified, data is read from stdin.")
	parser.add_argument("--output", "-o", help="Filepath to write labelled tweets to. If not specified, data is written to stdout.")
	
	args = parser.parse_args()
	
	if not os.path.isfile(args.cats):
		print("Error: File at '" + args.config + "' does not exist.")
		exit(1)
	if args.input and not os.path.isfile(args.input):
		print("Error: File at '" + args.input + "' does not exist.")
		exit(1)
	
	stream_in = sys.stdin
	stream_out = sys.stdout
	
	if args.input:
		stream_in = open(args.input, "r")
	if args.output:
		stream_out = open(args.output, "w")
	
	init_logging(None)
	
	if not args.cats:
		print("Error: No path to the categories file specified (--cats)")
		sys.exit(1)
	
	if not os.path.exists(args.cats):
		print(f"Error: No such file or directory {args.cats}")
		sys.exit(2)
	
	###############################################################################
	
	cats = CategoryCalculator(args.cats)
	
	labeller = TweetCatsLabeller(
		cats,
	)
	
	labeller.label(
		stream_in,
		stream_out
	)
	
	stream_out.close()


if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
