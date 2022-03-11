#!/usr/bin/env python3
# Ignore the above line, it's a Shebang: https://bash.cyberciti.biz/guide/Shebang


# SAMPLE CODE FOR 2022-03 SUSTAINABILITY HACKATHON, PROJECT 3
# Project 3: Live sentiment tracking during floods from social media data.
# University of Hull
# 
# This sample code shows how to load the pre-trained transformer model, and
# make a prediction for the first 10 tweets in a tweets.jsonl file.
# 
# Originally written by Lydia, PhD Student Researcher at the University of Hull
# aka Starbeamrainbowlabs <https://starbeamrainbowlabs.com>


import io
import json

import tensorflow as tf

# Custom layers the AI model uses. Ignore this.
from ai_layers.LayerPositionEmbedding import LayerPositionEmbedding
from ai_layers.LayerTransformerBlock import LayerTransformerBlock

# Converts input text into something the AI model can understand.
from glove.glove import GloVe

# Helper functions to convert class indexes (e.g. 0, 1) to class names (e.g. positive, negative)
# These aren't used in this example, but you might find them useful.
from cats import index2name, name2index

###
## 0: Preamble
###

# Change these filepaths to match your own filesystem.
# Note in particular the "g25" number, which may also be "g50", "g100", or "g200"
# depending on which model you are using.
# WARNING: The "g200" model requires at least 20GB of RAM!
filepath_checkpoint = "transformer-g25/transformer_checkpoint_g25.hdf5"
filepath_glove = "glove.twitter.27B.25d.txt"
filepath_tweets = "dataset/tweets.jsonl"



###
## 2: Load the data
###

glove = GloVe(filepath_glove)
model = tf.keras.models.load_model(filepath_checkpoint, custom_objects={
    # Tell Tensorflow about our custom layers so that it can deserialise models that use them
    "LayerPositionEmbedding": LayerPositionEmbedding,
    "LayerTransformerBlock": LayerTransformerBlock
})

# Load the first 10 tweets in
handle_tweets = io.open(filepath_tweets, "r")
tweets = []
i = 0
for line in handle_tweets:
    if i >= 10: # Only load the first 10 tweets for testing purposes
        break
    obj = json.loads(line)
    text = obj["text"].strip()
    
    # Use GloVe to convert the tweet text into a format the model understands
    text_embedding = glove.embeddings(text, length=100) # The original model was trained with a length of 100
    
    tweets.append(text_embedding)
    i = i + 1

###
## 3: Make some predictions
###

# The model was trained with a batch size of 64. If you can, pass in multiples
# of 64 tweets, as the model will automatically split them up batches of 64.
predictions = model(
    tf.constant(tweets, dtype="float32")
)


###
## 3: Display the predictions
###
# This outputs predictions in tab-separated-values (e.g. for importing into a
# spreadsheet), but you can do whatever you like here
# Note that even though you'll see lots of other random things in the console,
# these are all on the standard error, not the standard output.
print("item_number\tpositive\tnegative")
i = 0
for item in predictions:
    print(f"{i}\t{item[0]}\t{item[1]}")
    i += 1
