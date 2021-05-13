#!/usr/bin/env python3
import sys

from lib.glove.glove import GloVe

if len(sys.argv) < 3:
    print("""./glove.py
This script handles preprocessing text and convertin it to pretrained GloVe embeddings.
It is intended to be imported as a class, but a CLI is provided for convenience.

Usage:
path/to/glove.py <gloveloc> <text> [<mode=tweetvision>]

<gloveloc>	The path to the pretrained GloVe word vectors text file.
<text>		The text to process.
<mode>		Optional. The operation mode - defaults to tweetvision, which
            displays what the model will see the text as an array of
            strings. Specify "embeddings" here (without quotes) to display
            the actual word vectors instead.
""")
    exit(0)

gloveloc = sys.argv[1]
text = sys.argv[2]
mode = sys.argv[3] if len(sys.argv) >= 4 else "tweetvision"

glove = GloVe(gloveloc)

result = glove.tweetvision(text) if mode == "tweetvision" else glove.embeddings(text)

print()
print("\nINPUT:")
print(text)
print("\nOUTPUT:")
if mode == "tweetvision":
    print(result)
else:
    for i, item in enumerate(result):
        print(item)
