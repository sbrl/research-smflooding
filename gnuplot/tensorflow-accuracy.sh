#!/usr/bin/env bash

script_root="$(dirname "$(readlink -f "$0")")";

temp_dir="$(mktemp --tmpdir -d "gnuplot-XXXXXXX")";

on_exit() {
	rm -rf "${temp_dir}";
}
trap on_exit EXIT;

metrics_filename="${1}";
graph_title="${2} - Accuracy";

if [[ -z "${1}" ]] || [[ -z "${2}" ]]; then
	echo "Usage:" >&2;
	echo "    path/to/tensorflow-accuracy.sh <path/to/metrics.tsv> <graph_title>" >&2;
	exit
fi

output_filename="${metrics_filename%.*}-acc.png";

gnuplot -e "graph_title='${graph_title}'; data_filename='${metrics_filename}'" "${script_root}/tensorflow-accuracy.plt" >"${output_filename}";
