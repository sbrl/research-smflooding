#!/usr/bin/env python3
# TODO read in list of metrics.tsv files, compare 2 models as in the spreadsheet we're using etc

import sys
import os
import re

from loguru import logger
import pandas as pd


##############################################################

# Lifted from src/lib/polyfills/env.py

SYM_RAISE_EXCEPTION = "______sbrl_env_raise_exception"
envs_read = []


def env_read(name, type_class, default=SYM_RAISE_EXCEPTION):
	if name not in os.environ:
		if type_class == bool and default == SYM_RAISE_EXCEPTION:
			default = False
		if default == SYM_RAISE_EXCEPTION:
			raise Exception(f"Error: Environment variable {
							name} does not exist")
		envs_read.append([name, default, True])
		return default

	result = os.environ[name]
	if type_class == bool:
		result = False if default == True else True
	else:
		result = type_class(result)

	envs_read.append([name, result, False])
	return result

##############################################################


def read_tsv(filepath):
	if os.path.basename(filepath) != "metrics.tsv":
		logger.warning("WARNING: filename isn't metrics.tsv!")

	dirname = os.path.basename(os.path.dirname(filepath))

	return pd.read_csv(filepath, sep="\t")


def get_shortcode(dirname):
	matches = regex_shortcode.findall(dirname)
	if len(matches) < 1:
		return dirname

	return matches[0]

def get_cols(df, filter_substr):
	return [str(col) for col in df.columns if str(col).find(filter_substr) != -1]

##############################################################

# env var / cli arg parsing


if len(sys.argv) <= 1:
	print("""
This script produces graphs on metrics for a directory containing a series of clip-classifier experiments. It requires metrics.tsv to work.

Usage:
	clip_classifier_abl-compgraph.py <dirpath>
""")
	exit(0)

# THen search dirpath etc. regex for etracting shortcode?
dirpath = sys.argv[1]  # skip script path

REGEX_SHORTCODE = env_read("REGEX_SHORTCODE", str, "abl-[^_]+")


##############################################################


regex_shortcode = re.compile(REGEX_SHORTCODE)

logger.info(f"DEBUG dirpath: {dirpath}")

graphs = ["accuracy", ""]

# Read em all in
models = dict()
shortcodes = []
for dirpath_inner in os.scandir(dirpath):
	if not dirpath_inner.is_dir():
		continue
	dirname = str(os.path.basename(dirpath_inner))
	shortcode = get_shortcode(dirname)
	shortcodes.append(shortcode)
	models[shortcode] = read_tsv(os.path.join(dirpath_inner, "metrics.tsv"))

# Throw around some columns etc
result = None
i = 0
for shortcode, df in models.items():
	df.drop(axis=1, labels=get_cols(df, "batches"), inplace=True)
	df.columns = [col if col == "epoch" else f"{shortcode}_{col}" for col in df.columns]
	df.dropna(axis=1, how="all", inplace=True)
	if i > 0:
		df.drop(axis=1, labels="epoch")
		result = pd.merge(result, df, how="right")
	else:
		result = df
	
	i += 1


# print("DEBUG\n", models)
# print("DEBUG\n", result)

# print("DEBUG:cols/acc", get_cols(result, "accuracy"))

def do_plot(df, field):
	logger.info(f"Plotting for field {field}")
	ax = df.plot(
		x="epoch",
		y=get_cols(result, field),
		xlabel="epochs",
		ylabel=field,
		figsize=(15,10),
		title=f"{field}: {" vs ".join(shortcodes)}"
	)
	ax.figure.savefig(os.path.join(dirpath, f"{field}.png"))


do_plot(result, "accuracy")
do_plot(result, "loss")

logger.success(f"Plots saved to {dirpath}")