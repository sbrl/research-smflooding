import io
import json
from functools import partial
import numpy
import sys

def tweets_data_simple(filepath_input):
    reader = io.open(filepath_input, "r")
    sys.stdout.write("\n")
    result = []
    i = 0
    for line in reader:
        obj = json.loads(line)
        result.append(obj["text"].strip())
        i = i + 1
        
        if i % 10000 == 0:
            sys.stdout.write(f"\rtweets_data_simple: loaded {i}")
    
    sys.stdout.write(" done\n")
    return result
