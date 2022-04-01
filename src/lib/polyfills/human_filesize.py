import math

def human_filesize(nbytes, decimals = 2):
	sz = ["b", "kib", "mib", "gib", "tib", "pib", "eib", "yib", "zib"]
	factor = math.floor((len(str(nbytes)) - 1) / 3)
	result = round(nbytes / (1024 ** factor), decimals)
	return str(result) + sz[factor]
