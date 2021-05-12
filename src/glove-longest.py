#!/usr/bin/env python3
import io
from pathlib import Path
import argparse
import logging
import json

import tensorflow as tf

from lib.glove.glove import GloVe

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="This program calculates the longest word embedding in a list of tweets.")
parser.add_argument("--glove", "-g", help="Filepath to the pretrained GloVe word vectors to load.")
parser.add_argument("input", help="The input tweets jsonl file to scan.")


args = parser.parse_args()

if not Path(args.file).is_file():
    print("Error: File at '" + args.file + "' does not exist.")
    exit(1)

###############################################################################

glove = GloVe(args.glove)
longest = 0

handle = io.open(args.input, "r")
for i, line in enumerate(handle):
    obj = json.loads(line)
    result = glove.convert(obj.text)
    
    if len(result) > longest:
        longest = len(result)
        logging.info(f"Tweet #{i} has length of {longest}")


# print(tf.constant(glove.convert(args.input_string)))
