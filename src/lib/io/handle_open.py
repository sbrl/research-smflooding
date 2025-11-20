import io
import gzip


try:
	import compression.zstd as zstd
except ImportError:
	print("handle_open: Not Python 3.14 or above; zstd decompression is not supported --@sbrl")

def handle_open(filepath, mode, force_textwrite_gzip=True):
	if mode == "w" and mode.endswith(".gz") and force_textwrite_gzip:
		mode = "wt"
	
	if filepath.endswith(".gz"):
		return gzip.open(filepath, mode)
	elif filepath.endswith(".zst") or filepath.endswith(".zstd") and zstd:
		return zstd.open(filepath, mode)
	else:
		return io.open(filepath, mode)