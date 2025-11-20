#!/usr/bin/env python3
import os
import sys
import time
import json

from loguru import logger
import umap
import umap.plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datashader as ds
import colorcet

from lib.io.handle_open import handle_open

if "--help" in sys.argv:
    print("""Embeddings → UMAP convertificator and plotificator 9001
	By Starbeamrainbowlabs

Usage:
	[ENV_VAR=value ....] path/to/umapify.py

Environment variables:
	INPUT   	The path to the input jsonl file
	OUTPUT  	The path to the output tsv file. Will have DIM+1 columns in the form '[ label, dim_1, dim_2, ... dim_x ] @ tsv'. A sister file will be placed with the file extension .png with a Cool Plot™
	DIM			The number of output dimensions to UMAP to.

Extra info:
	The input file should be in a .jsonl file, with 1 object per line. The following properties are recognised as the embedding vector and the associated label:
	EMBEDDING: embed, clip
	LABEL: label, filepath

""")
    exit(0)

# ███████ ███    ██ ██    ██
# ██      ████   ██ ██    ██
# █████   ██ ██  ██ ██    ██
# ██      ██  ██ ██  ██  ██
# ███████ ██   ████   ████

FILEPATH_INPUT = os.environ["INPUT"] if "INPUT" in os.environ else None
FILEPATH_OUTPUT = os.environ["OUTPUT"] if "OUTPUT" in os.environ else None
DIM = int(os.environ["DIM"]) if "DIM" in os.environ else 2
INPUT_FORMAT = "single" if "INPUT_SINGLE" in os.environ else "double"

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

start = time.time()


def find_properties(obj, props: list):
	for prop in props:
		if prop in obj:
			return obj[prop]
	return None


labels: list = []
embed: list = []

i_read = 0
with handle_open(FILEPATH_INPUT, "r") as handle:
	for line in handle:
		i_read += 1
		if type(line) is bytes:
			line: str = line.decode()
		
		if not line.strip():
			continue # skip empty lines
		
		obj = json.loads(line)
		
		label = find_properties(obj, ["label", "filepath"])
		if label is None:
			raise Exception(f"Error: For line {i_read} found None for label")
		embed = find_properties(obj, ["embed", "clip"])
		if embed is None:
			raise Exception(f"Error: For line {i_read} found None for embed")
		
		labels.append(label)
		embed.append(embed)
		
		if i_read % 1000 == 0:
			sys.stderr.write(f"Reading lines: {i_read} items read so far\r")
		

logger.info(f"{i_read} items read in {round(time.time() - start, 3)}s")


embed_np = np.array(embed)

# ██    ██ ███    ███  █████  ██████  
# ██    ██ ████  ████ ██   ██ ██   ██ 
# ██    ██ ██ ████ ██ ███████ ██████  
# ██    ██ ██  ██  ██ ██   ██ ██      
#  ██████  ██      ██ ██   ██ ██      

logger.info("UMAPing...")
umapped = umap.UMAP(
	min_dist=0.05,
	n_components=DIM
).fit_transform(embed_np)
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
		ds.utils.export_image( # it does too exist....!
			result,
			filepath_target
		)
		print("canvas", canvas, "points", points, "result", result)
		logger.info(f"Written plot with 2 dimensions to {filepath_target}.png")
	else:
		logger.info(f"Warning: Not exporting a plot, since a dim of {dim} is not supported (supported values: 2).")

def save_tsv(filepath_target, umapped, labels):
	logger.info("Writing tsv")
	with handle_open(filepath_target, "w") as handle:
		print("REDUCED VECTORS\n", umapped[0:10])
		print("LABELS\n", labels[0:10])
		# df_points = pd.DataFrame(umapped)
		# df_labels = pd.DataFrame(words)
		# df_labels.columns = ["word"]
		
		rows = [ [ row[0], *row[1] ] for row in zip(labels, umapped) ]
		
		for row in rows:
			payload = "\t".join([str(item) for item in row]) + "\n"
			handle.write(payload.encode() if filepath_target.endswith(".gz") else payload)
	
	logger.info(f"Written values to {filepath_target}")

plot(
	filepath_target=filepath_output_image,
	umapped=umapped,
	dim=DIM
)
save_tsv(FILEPATH_OUTPUT, umapped, labels)
