#!/usr/bin/env python3
import os
import sys
import gzip
import time

from loguru import logger
import umap
import umap.plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datashader as ds
import colorcet

from lib.io.handle_open import handle_open
from lib.glove.glove import GloVe

if "--help" in sys.argv:
	print("""Wordlist → UMAP convertificator and plotificator 9000
	By Starbeamrainbowlabs

Usage:
	[ENV_VAR=value ....] path/to/umapify.py

Environment variables:
	INPUT   The path to the wordlist file. See the command below for more info.
	OUTPUT  The path to the output tsv file. Will have DIM+1 columns in the form '[ word, dim_1, dim_2, ... dim_x ] @ tsv'. A sister file will be placed with the file extension .png with a Cool Plot™
	DIM		The number of output dimensions to UMAP to.
	GLOVE	The filepath to the glove model to use.

Extra info:
	Make a wordlist from a tweet .jsonl file like so:
		 jq --raw-output -c .text tweets-all-new-20220117.jsonl | fmt -w1 | sort -n | uniq -c | less | sort -n >UMAP-tweets-all-new-20220117-wordlist.txt
		 
		The output is in the form "^[0-9]+\\t.*$", in which [0-9]+ is the frequency (not used), and .* is the word itself.
""")
	exit(0)

# ███████ ███    ██ ██    ██
# ██      ████   ██ ██    ██
# █████   ██ ██  ██ ██    ██
# ██      ██  ██ ██  ██  ██
# ███████ ██   ████   ████

FILEPATH_INPUT = os.environ["INPUT"] if "INPUT" in os.environ else None
FILEPATH_OUTPUT = os.environ["OUTPUT"] if "OUTPUT" in os.environ else None
FILEPATH_GLOVE = os.environ["GLOVE"] if "GLOVE" in os.environ else None
DIM = int(os.environ["DIM"]) if "DIM" in os.environ else 2
FILEPATH_STOPWORDS = os.environ["STOPWORDS"] if "STOPWORDS" in os.environ else None

filepath_output_image = os.path.join(
	os.path.dirname(FILEPATH_OUTPUT),
	os.path.splitext(os.path.basename(
		FILEPATH_OUTPUT.replace(".gz", "")
	))[0] # The .png is added automatically by datashader
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

if FILEPATH_INPUT is None or not os.path.exists(FILEPATH_INPUT):
	raise Exception(f"Error: No file found at '{FILEPATH_INPUT}'. Either it doesn't exist, or you don't have permission to read it.")
if FILEPATH_GLOVE is None or not os.path.exists(FILEPATH_GLOVE):
	raise Exception(f"Error: No file found at '{FILEPATH_GLOVE}'. Either it doesn't exist, or you don't have permission to read it.")
if FILEPATH_OUTPUT is None or FILEPATH_OUTPUT == "":
	raise Exception(f"Error: No output filepath specified.")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

logger.info("Wordlist → UMAP convertificator and plotificator 9000")
for env_name in ["FILEPATH_INPUT", "FILEPATH_OUTPUT", "filepath_output_image", "FILEPATH_GLOVE", "FILEPATH_STOPWORDS", "DIM"]:
	logger.info(f"> {env_name} {str(globals()[env_name])}")


# ██████   █████  ████████  █████
# ██   ██ ██   ██    ██    ██   ██
# ██   ██ ███████    ██    ███████
# ██   ██ ██   ██    ██    ██   ██
# ██████  ██   ██    ██    ██   ██

glove = GloVe(FILEPATH_GLOVE)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

sys.stderr.write("Loading stop words...")
start = time.time()
stop_words = None
if FILEPATH_STOPWORDS is not None:
	with handle_open(FILEPATH_STOPWORDS, "r") as handle:
		stop_words = []
		for line in handle:
			if type(line) is bytes:
				line = line.decode()
			stop_word = line.rstrip("\n")
			if stop_word == "":
				continue
			stop_words.append(stop_word)
stop_words = set(stop_words)
sys.stderr.write(f" loaded {len(stop_words)} stop words in {round(time.time() - start, 3)}s.\n")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

start = time.time()
words = []
with handle_open(FILEPATH_INPUT, "r") as handle:
	stop_words_skipped = 0
	words_read = 0
	for line in handle:
		if type(line) is bytes:
			line = line.decode()
		row = line.split("\t", maxsplit=1)
		if row == "" or len(row) < 2:
			continue
		try:
			row[0] = int(row[0])
		except:
			continue
		row[1] = row[1].rstrip("\n")
		if stop_words is not None and row[1] in stop_words:
			stop_words_skipped += 1
			continue
		words.append(row[1])
		words_read += 1
		if words_read % 1000 == 0:
			sys.stderr.write(f"Reading words: {words_read} words read so far\r")
		

logger.info(f"{len(words)} read in {round(time.time() - start, 3)}s, {stop_words_skipped} stop words skipped")


#  ██████  ██       ██████  ██    ██ ███████ 
# ██       ██      ██    ██ ██    ██ ██      
# ██   ███ ██      ██    ██ ██    ██ █████   
# ██    ██ ██      ██    ██  ██  ██  ██      
#  ██████  ███████  ██████    ████   ███████ 

def flatten(values):
    values = list(values)
    output = []

    for i in range(len(values)):
        for j in range(len(values[i])):
            output.append(values[i][j])

    return output


words_glove = [[row, glove.lookup(row)] for row in words]
words_glove = list(filter(lambda row: len(row) == 2 and row[1] is not None, words_glove))
# print("UNIQ", set([ len(row) for row in words_glove ]))
# print(len(words_glove))
# print(words_glove[0:10])

logger.info(f"{len(words_glove)} words mapped using GloVe")

words_glove_embed = np.array([ item[1] for item in words_glove ])
words_glove_source = [ item[0] for item in words_glove ]

# ██    ██ ███    ███  █████  ██████  
# ██    ██ ████  ████ ██   ██ ██   ██ 
# ██    ██ ██ ████ ██ ███████ ██████  
# ██    ██ ██  ██  ██ ██   ██ ██      
#  ██████  ██      ██ ██   ██ ██      

logger.info("UMAPing...")
umapped = umap.UMAP(
	min_dist=0.05,
	n_components=DIM
).fit_transform(words_glove_embed)
logger.info("UMAP conversion complete")



# dim_reducer = umap.UMAP(
# 	min_dist=0.05  # default: 0.1
# ).fit(words_glove)


# ██████  ██       ██████  ████████ ████████ ██ ███    ██  ██████
# ██   ██ ██      ██    ██    ██       ██    ██ ████   ██ ██
# ██████  ██      ██    ██    ██       ██    ██ ██ ██  ██ ██   ███
# ██      ██      ██    ██    ██       ██    ██ ██  ██ ██ ██    ██
# ██      ███████  ██████     ██       ██    ██ ██   ████  ██████

def plot(filepath_target, umapped, dim):
	logger.info("Plotting")
	if dim == 2:
		df = pd.DataFrame(umapped)
		df.columns = ["x", "y"]
		
		print(df)
		
		canvas = ds.Canvas(plot_width=850, plot_height=850)
		points = canvas.points(df, "x", "y")
		result = ds.tf.set_background(ds.tf.shade(points), color="white")
		ds.utils.export_image(
			result,
			filepath_target
		)
		print("canvas", canvas, "points", points, "result", result)
		logger.info(f"Written plot with 2 dimensions to {filepath_target}.png")
	else:
		logger.info(f"Warning: Not exporting a plot, since a dim of {dim} is not supported (supported values: 2).")

def save_tsv(filepath_target, umapped, words):
	logger.info("Writing tsv")
	with handle_open(filepath_target, "w") as handle:
		print(umapped[0:10])
		print(words[0:10])
		# df_points = pd.DataFrame(umapped)
		# df_labels = pd.DataFrame(words)
		# df_labels.columns = ["word"]
		
		rows = [ [ row[0], *row[1] ] for row in zip(words, umapped) ]
		
		for row in rows:
			payload = "\t".join([str(item) for item in row]) + "\n"
			handle.write(payload.encode() if filepath_target.endswith(".gz") else payload)
	
	logger.info(f"Written values to {filepath_target}")

plot(
	filepath_target=filepath_output_image,
	umapped=umapped,
	dim=DIM
)
save_tsv(FILEPATH_OUTPUT, umapped, words_glove_source)