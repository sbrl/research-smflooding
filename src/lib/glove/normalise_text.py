"""
Script for preprocessing tweets by Romain Paulus.
Translation of Ruby script to create features for GloVe vectors for Twitter data.

with small modifications by Jeffrey Pennington
with translation to Python by Motoki Wu
updated to Python 3 by Phil Pope (@ppope)
with bugfixes and improvements by Starbeamrainbowlabs (@sbrl)
	• Tidy up code with help of linters
	• Add spaces surrounding punctuation and <token_blocks>
	• Limit runs of whitespace to a single space
	• Transliterate ’ to '

Ref original Ruby source http://nlp.stanford.edu/projects/glove/preprocess-twitter.rb
Ref https://gist.github.com/ppope/0ff9fa359fb850ecf74d061f3072633a
"""

import sys
import re

FLAGS = re.MULTILINE | re.DOTALL

# CHANGED 2021-11 after 2nd rerun: Handle ’ → ' and putting spaces before / after <allcaps>, [, and ]

def hashtag(text):
	"""Handles hashtags."""
	text = text.group()
	hashtag_body = text[1:]
	if hashtag_body.isupper():
		result = " {} ".format(hashtag_body.lower())
	else:
		# Was re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS)
		# ref https://stackoverflow.com/a/2277363/1460422
		result = " ".join(["<hashtag>"] + re.findall("^[a-z]+|[A-Z][^A-Z]*", hashtag_body, flags=FLAGS))
	return result


def allcaps(text):
	"""Handles all caps text."""
	text = text.group()
	return text.lower() + " <allcaps> "


# Convenience function to reduce repetition
def re_sub(pattern, repl, text):
	return re.sub(pattern, repl, text, flags=FLAGS)

def normalise(text):
	"""
	Preprocesses the given input text to make it suitable for GloVe.
	This is the main function you want to import.
	:param	str		text: The text to normalise.
	"""
	# Different regex parts for smiley faces
	eyes = r"[8:=;]"
	nose = r"['`\-]?"

	text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ", text)
	text = re_sub(r"@\S+", " <user> ", text)
	text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smile> ", text)
	text = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ", text)
	text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ", text)
	text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ", text)
	text = re_sub(r"/", " / ", text)
	text = re_sub(r"<3", " <heart> ", text)
	text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ", text)
	text = re_sub(r"#\S+", hashtag, text)
	text = re_sub(r"([!?.]){2,}", r" \1 <repeat> ", text)
	text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r" \1\2 <elong> ", text)
	
	# -- I just don't understand why the Ruby script adds <allcaps> to everything so I limited the selection.
	# text = re_sub(r"([^a-z0-9()<>'`\-]){2,}", allcaps)
	text = re_sub(r"([A-Z]){2,}", allcaps, text)
	
	# Added by @sbrl
	
	text = re_sub(r"’", "'", text)
	text = re_sub(r"([!?:;.,\[\]])", r" \1 ", text)	# Spaces around punctuation
	
	text = re_sub(r"\s+", r" ", text)	# Limit runs of whitespace to a single space
	
	return text.lower()


if __name__ == '__main__':
	_, text = sys.argv
	if text == "test":
		text = "I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!"
	tokens = normalise(text)
	print(tokens)
