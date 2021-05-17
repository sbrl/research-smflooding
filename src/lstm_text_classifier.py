#!/usr/bin/env python3
import io
from pathlib import Path
import argparse
import logging
import json
# import tensorflow as tf



def main():
    """Main entrypoint."""
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="This program calculates trains a tweet classification model.")
    parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.")

    
    args = parser.parse_args()

    if not Path(args.tweets_jsonl).is_file():
        print("Error: File at '" + args.tweets_jsonl + "' does not exist.")
        exit(1)
    
    ###############################################################################
    
    # args.glove
    # TODO: Import model, write and instantiate instance of data preprocessing class.
    # Investigate keras sequence for dataset management - apparently the batch size is specified there.


    # print(tf.constant(glove.convert(args.input_string)))

if __name__ == "__main__":
    main()
else:
    print("This script must be run directly. It cannot be imported.")
    exit(1)
