#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
import tensorflow as tf

from lib.glove.glove import GloVe

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="This program tests GloVe SQLite databases.")
parser.add_argument("file", help="Filepath to the pretrained GloVe word vectos to load.")
parser.add_argument("--input-string", "-i", help="The input string to tokenise.")


args = parser.parse_args()

if not Path(args.file).is_file():
    print("Error: File at '" + args.file + "' does not exist.")
    exit(1)

###############################################################################

glove = GloVe(args.file)

print(tf.constant(glove.convert(args.input_string)))
