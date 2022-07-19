#!/usr/bin/env node
"use strict";

import { plot } from 'nodeplotlib';

const sequence_length = 100;
const dim_size = 20;

// Ref https://dev.to/ycmjason/how-to-create-range-in-javascript-539i
function range(start, end) {
	return (new Array(end - start + 1)).fill(undefined).map((_, i) => i + start);
}


const vectors = [];
for (let i_seq = 0; i_seq < sequence_length; i_seq++) {
	const row = [];
	for (let i_dim = 0; i_dim < dim_size; i_dim++) {
		if(i_dim % 2 == 0) // PE_{(i_{seq}, 2i_{dim})} = sin(\frac{i_{seq}}{10 000^{2i_{dim}/dim}})
			row.push(Math.sin(i_seq / (10000 ** ((2*i_dim)/dim_size))));
		else // PE_{(i_{seq},2i_{dim}+1)} = cos(\frac{i_{seq}}{10 000^{2i_{dim}/dim}})
			row.push(Math.cos(i_seq / (10000 ** ((2*i_dim)/dim_size))));
	}
	vectors.push(row);
}

plot([
	{
		y: range(0, sequence_length),
		z: vectors,
		type: "heatmap",
		name: `Transformer positional encoding signal`
	}
], {
	xaxis: { title: "embedding dimension" },
	yaxis: { title: "sequence dimension" }
})