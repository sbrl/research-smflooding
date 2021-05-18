#!/usr/bin/env python3
import io
from pathlib import Path
import argparse
import logging
import json
# import tensorflow as tf

from ..io.settings import settings_get, settings_load
from lib.data.TweetsData import TweetsData
from lib.ai.LSTMTweetClassifier import LSTMTweetClassifier


def main():
    """Main entrypoint."""
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="This program calculates trains a tweet classification model.")
    parser.add_argument("--config", "-c", help="Filepath to the TOML config file to load.", required=True)
    parser.add_argument("--output", "-o", help="Path to output directory to write output to (will be automatically created if it doesn't exist)", required=True)
    
    args = parser.parse_args()
    
    if not Path(args.tweets_jsonl).is_file():
        print("Error: File at '" + args.tweets_jsonl + "' does not exist.")
        exit(1)
    
    settings_load(args.config)
    
    settings = settings_get()
    settings.output = args.output
    
    ###############################################################################
    
    dataset = TweetsData()
    ai = LSTMTweetClassifier()
    
    # TODO: Train the model here
    # print(tf.constant(glove.convert(args.input_string)))


if __name__ == "__main__":
    main()
else:
    print("This script must be run directly. It cannot be imported.")
    exit(1)
