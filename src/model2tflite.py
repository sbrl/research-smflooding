#!/usr/bin/env python3

import os

import tensorflow as tf

INPUT = os.environ["INPUT"]		# Input Keras model
OUTPUT = os.environ["OUTPUT"]	# Output filepath, should end in .tflite

from lib.ai.LayerPositionEmbedding import LayerPositionEmbedding
from lib.ai.LayerTransformerBlock import LayerTransformerBlock

model = tf.keras.models.load_model(INPUT, custom_objects={
    # Tell Tensorflow about our custom layers so that it can deserialize models that use them
	"LayerPositionEmbedding": LayerPositionEmbedding,
	"LayerTransformerBlock": LayerTransformerBlock
})

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(OUTPUT, "wb") as handle:
	handle.write(tflite_model)
