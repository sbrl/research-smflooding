#!/usr/bin/env bash

target_file="${1}";

if [[ -z "${target_file}" ]]; then
	echo "glove2sqlite.sh
This script converts pre-trained word vectors to an SQLite3 database.

Usage:
	glove2sqlite.sh <file>

Download and extract the pre-trained word vectors from here: https://nlp.stanford.edu/projects/glove/

To download:
	curl -O <url>

Then, to extract:

unzip path/to/file.zip

";
	exit 0;
fi

if [[ ! -r "${target_file}" ]]; then
	echo "Error: Can't read from '${target_file}'. Have you checked the spelling and file permissions?" >&2;
	exit 1;
fi

db_file="${target_file%.*}.sqlite3";

if [[ -e "${db_file}" ]]; then
	echo "Error: A file already exists at '${db_file}', so not creating an SQLite3 database to avoid overwriting data. Delete or move it out of the way and then try running this script again." >&2;
	exit 2;
fi

###############################################################################

temp_dir="$(mktemp --tmpdir -d "glove2sqlite-XXXXXXX")";
on_exit() {
	rm -rf "${temp_dir}";
}
trap on_exit EXIT;

log_msg() {
	echo "${SECONDS} >>> $*";
}

###############################################################################

log_msg "Creating database";
touch "${db_file}";
# sqlite3 "${db_file}" 'CREATE TABLE IF NOT EXISTS data (key TEXT, value TEXT)';

log_msg "Importing data";
awk '{ gsub(/^<|>$/, "", $1); sub(/ /, "\t", $0); print($0) }' <"${target_file}" | sqlite3 "${db_file}" ".mode tabs
.import '|cat -' data.tsv";

log_msg "Complete";
