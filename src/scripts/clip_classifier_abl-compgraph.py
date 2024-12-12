
# TODO read in list of metrics.tsv files, compare 2 models as in the spreadsheet we're using etc

import os

from loguru import logger
import pandas as pd

def read_tsv(filepath):
	if os.path.basename(filepath) != "metrics.tsv":
		logger.warning("WARNING: filename isn't metrics.tsv!")
	
	dirname = os.path.basename(os.path.dirname(filepath))
	
	return pd.read_csv(filepath, sep="\t")


# TODO get shortcode → filepath pairs somehow


