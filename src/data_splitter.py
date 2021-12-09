#!/usr/bin/env python3
import io
import os
import argparse
from loguru import logger
import json
# import tensorflow as tf

from lib.data.CategoryCalculator import CategoryCalculator


def main():
	"""Main entrypoint."""
	
	parser = argparse.ArgumentParser(description="This program splits a LABELLED tweets JSONL file into multiple subfiles based on the attached predicted labels.")
	parser.add_argument("--input", "-i", help="Path to input tweets jsonl file to read.", required=True)
	parser.add_argument("--cats", "-c", help="Filepath to the categories file to load.", required=True)
	parser.add_argument("--output", "-o", help="Path to output directory to write data to (will be automatically created if it doesn't exist)", required=True)
	
	args = parser.parse_args()
	
	if not os.path.exists(args.cats):
		print("Error: The specified categories file doesn't exist.")
		exit(1)
	if not os.path.exists(args.input):
		print("Error: The input tweets jsonl file doesn't exist.")
		exit(1)
	if not os.path.exists(args.output):
		os.makedirs(args.output)
	
	###############################################################################
	
	cats = CategoryCalculator(args.cats)
	split(args.input, args.output, cats)
	


def split(filepath_input, dir_output, cats):
	"""Splits the given input into multiple separate files."""
	reader = io.open(filepath_input, "r")
	writers = { }
	
	for line in reader:
		obj = json.loads(line)
		text = obj["text"].strip()
		
		next_cat = cats.get_category_name(text)
		
		if next_cat is None:
			next_cat = "none"
		
		if next_cat not in writers:
			writers[next_cat] = io.open(os.path.join(dir_output, f"{next_cat}.jsonl"), "w")
		
		writers[next_cat].write(line)
	
	for cat_name in writers:
		writers[cat_name].close()


if __name__ == "__main__":
	main()
else:
	print("This script must be run directly. It cannot be imported.")
	exit(1)
