#!/usr/bin/env python3
import io
from pathlib import Path
import argparse
from loguru import logger
import json
# import tensorflow as tf

from lib.glove.glove import GloVe


def main():
    """Main entrypoint."""
    
    parser = argparse.ArgumentParser(description="This program calculates the longest word embedding in a list of tweets.")
    parser.add_argument("--glove", "-g", help="Filepath to the pretrained GloVe word vectors to load.")
    parser.add_argument("tweets_jsonl", help="The input tweets jsonl file to scan.")


    args = parser.parse_args()

    if not Path(args.tweets_jsonl).is_file():
        print("Error: File at '" + args.tweets_jsonl + "' does not exist.")
        exit(1)
    
    ###############################################################################

    glove = GloVe(args.glove)
    longest = 0

    handle = io.open(args.tweets_jsonl, "r")
    for i, line in enumerate(handle):
        obj = json.loads(line)
        result = glove.tweetvision(obj["text"])
        
        if len(result) > longest:
            longest = len(result)
            logger.info(f"\n\n\n\nTweet #{i} has length of {longest}:")
            logger.info("INPUT:")
            logger.info(obj["text"])
            logger.info("\nOUTPUT:")
            logger.info(result)


    # print(tf.constant(glove.convert(args.input_string)))

if __name__ == "__main__":
    main()
else:
    print("This script must be run directly. It cannot be imported.")
    exit(1)
