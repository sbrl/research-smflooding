#!/usr/bin/env bash

OLD_PWD="${PWD}";

# Make sure the current directory is the location of this script to simplify matters
cd "$(dirname "$(readlink -f "$0")")" || { echo "Error: Failed to cd to script directory" >&2; exit 1; };

lantern_path="./lantern-build-engine";
# Check out the lantern git submodule if needed
if [ ! -f "${lantern_path}/lantern.sh" ]; then git submodule update --init "${lantern_path}"; fi

#shellcheck disable=SC1090
source "${lantern_path}/lantern.sh";

###############################################################################





###############################################################################

#  ██████ ██      ██
# ██      ██      ██
# ██      ██      ██
# ██      ██      ██
#  ██████ ███████ ██

check_command grep;
check_command python;
check_command jq;
# check_command node;
# check_command npm;


export mode="MAIN"; # MAIN, FILE

while test "$#" -gt 0
do
	case "$1" in
		--help)
			# TODO: Finish this help message
			# Ref https://discord.com/channels/364428045093699594/364434363472936974/896083536933695508
			echo "Social-Media-Sentiment"
			echo "    by Starbeamrainbowlabs <feedback@starbeamrainbowlabs.com>"
			echo "This superscript provides a convenient interface by which all the various entrypoints in this repository can be neatly accessed.";
			echo "";
			echo "Note: You need to have your tweets already downloaded to use this repository. For this, use twitter-academic-downloader:";
			echo "    <https://www.npmjs.com/package/twitter-academic-downloader>";
			echo "";
			echo "To get help on a specific subcommand, do \"path/to/main.sh <subcommand> --help\".";
			echo "";
			echo "Subcommands:"
			echo "    train           Train a new tweet text classifier";
			echo "    train-image     Train a new image classifier [requires a tweet text classifier to have been trained first]";
			echo "    label-tweets    Given a trained tweet text classifier, label a tweets JSONL file.";
			echo "    label-images    Given a trained image classifier and a directory of images, outputs filenames and predicted classes as tab-separated values.";
			echo "";
			echo "Extra subcommands for diagnostics:";
			echo "(these are not guaranteed to have a stable CLI)";
			echo "    confusion       Renders a confusion matrix for a tweet text classifier.";
			echo "    glove-longest   Given a tweets JSONL file and a GloVe embeddings file, output the longest GloVe sequence length found.";
			echo "    split-labelled  Split a LABELLED tweets JSONL file into multiple sub-files based on the attached labels. See the label-tweets subcommand for labelling tweets.";
			echo "";
			echo "Options:";
			echo "    --help";
			echo "         Show this help message";
			exit 0;
			;;
		
		train-text)
			shift;
			./src/text_classifier.py "$@";
			;;
		
		train-image )
			shift;
			./src/image_classifier.py "$@";
			;;
		
		confusion )
			shift;
			./src/confusion_matrix.py "$@";
			;;
		glove-longest )
			shift;
			./src/glove_longest.py "$@";
			;;
		split-labelled )
			shift;
			./src/data_splitter.py "$@";
			;;
		
		label-tweets )
			shift;
			./src/label_tweets.py "$@";
			;;
		label-images )
			shift;
			./src/label_images.py "$@";
			;;
		
		# TODO: Add mroe subcommands here
	esac
	shift
done
