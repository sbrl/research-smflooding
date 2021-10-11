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
			echo "Subcommands:"
			echo "    train";
			echo 
			echo "Options:"
			echo "    --help"
			echo "         Show this help message"
			echo "    --dry-run"
			echo "         Do a dry run - don't actually move any files."
			exit 
			;;
		
		train-text)
			shift;
			./src/text_classifier.py "$@";
			;;
			
		train-image )
			shift;
			./src/image_classifier.py "$@";
			;;
		
		label-tweets )
			shift;
			./src/label_tweets.py "$@";
			;;
		
		# TODO: Add mroe subcommands here
	esac
	shift
done
