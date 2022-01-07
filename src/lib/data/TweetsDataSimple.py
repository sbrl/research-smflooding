import io
import json
from functools import partial
import numpy


def tweets_data_simple(filepath_input):
    reader = io.open(filepath_input, "r")
    
    result = []
    for line in reader:
        obj = json.loads(line)
        result.append(obj["text"].strip())
    
    return result
