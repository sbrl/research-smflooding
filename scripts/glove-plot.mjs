#!/usr/bin/env node
"use strict";

import fs from 'fs';

import { plot } from 'nodeplotlib';
import nexline from 'nexline';
import cossim from 'cos-similarity';

const words = [ "flood", "water", "rain", "unrelated" ];

const glove_location = "/mnt/research-data/main/glove/glove.twitter.27B.25d.txt";

let glove = {};

const reader = nexline({ input: fs.createReadStream(glove_location) });
let i = -1;
while (true) {
	i++;
	const line = await reader.next();
	if(line === null) break;
	if(line.trim() === "") continue;
	
	const parts = line.split(" ");
	glove[parts[0]] = parts.slice(1).map(value => parseFloat(value));
	
	if(i % 10000 === 0) process.stderr.write(`GloVe: Reading entry ${i}\r`);
}

const vectors = words.map(word => glove[word.toLowerCase()]);

const similarities = vectors.map(vector => {
	return vectors.map(other => cossim(vector, other));
});

console.log(`similarities`, similarities);

plot([
	{
		y: words,
		z: vectors,
		type: "heatmap",
		name: `GloVe d25 word vectors`
	}
])
plot([
	{
		x: words,
		y: words,
		z: similarities,
		type: "heatmap",
		title: `GloVe d25 word cosine similarities`,
		colorbar: {
			tickmode: "linear",
			tick0: 0,
			dtick: 0.1
		}
	}
]);
