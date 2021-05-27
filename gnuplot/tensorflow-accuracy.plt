set title graph_title font "Ubuntu,24"
set xlabel "Epochs" font "Ubuntu,20"
set ylabel "Accuracy" font "Ubuntu,20"
set key title "Legend" font "Ubuntu,20"
set grid

set yrange [0:1]
set datafile separator "\t"
set terminal png linewidth 3 size 1366,768
plot data_filename using 1:2 with lines title "Training", data_filename using 1:4 with lines title "Validation"
