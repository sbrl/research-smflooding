import io
import gzip


def handle_open(filepath, mode, force_textwrite_gzip=True):
	if mode == "w" and mode.endswith(".gz") and force_textwrite_gzip:
		mode = "wt"
	
	if filepath.endswith(".gz"):
		return gzip.open(filepath, mode)
	else:
		return io.open(filepath, mode)