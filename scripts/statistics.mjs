#!/usr/bin/env node
"use strict";


import fs from 'fs';

import { fOneScoreMacro as f1score } from 'data-science-js';

const filepath = process.argv[2];
const mode = process.env.OUTPUT_MODE ? process.env.OUTPUT_MODE : "csv";
const decimal_places = process.env.DECIMAL_PLACES ? parseInt(process.env.DECIMAL_PLACES) : 3;
const collapse_neutral_positive = process.env.COLLAPSE_NEUPOS ? true : false;

console.error(`Collapse neutral/positive is ${collapse_neutral_positive?"ON":"OFF"}`);

function normalise_sentiment(sentiment) {
	let result = sentiment.toLowerCase().replace(/slightly\s+/, "");
	switch(result) {
		case "neu":
			result = "neutral";
			break;
		case "pos":
			result = "positive";
			break;
		case "neg":
			result = "negative";
			break;
	}
	if(collapse_neutral_positive && result == "neutral")
		result = "positive";
	return result;
}

function calculate_stats(arr, contains_neutral = false) {
	const trans_table = {
		positive: 1,
		neutral: 0,
		negative: -1
	};
	
	// 1: Convert to numbers
	let numerical = arr.map(row => row.map(cell => {
		if(typeof cell !== "string" || typeof trans_table[cell] !== "number")
			console.log("trans_table failure | ROW", row, "CELL", cell);
		return trans_table[cell];
	}));
	
	if(!contains_neutral)
		numerical = numerical.filter(row => row[0] !== 0);
	
	// 2: Do calculation
	const result = f1score(
		numerical.map(row => row[0]),
		numerical.map(row => row[1])
	);
	
	// 3: Add in accuracy
	if(typeof result.accuracy !== "undefined")
		throw new Error(`Accuracy exists when it shouldn't`);
	if(contains_neutral)
		result.accuracy = arr.filter(row => row[0] === row[1]).length / arr.length;
	else
		result.accuracy = arr.filter(row => row[0] !== `neutral`)
			.filter(row => row[0] === row[1]).length / arr.length;
	
	return result;
}

let data = fs.readFileSync(filepath, "utf-8").split(`\n`)
	.filter(line => line.length > 0)
	.map(line => line.split(`\t`));

const header = data[0];
data = data.slice(1)
	.map(parts => header.reduce((obj, colname, i) => {
		obj[colname] = i > 0 ? normalise_sentiment(parts[i]) : parts[i];
		return obj;
	}, {}))


const data_noneutral = data.filter(obj => obj.sentiment_human !== "neutral");

const acc = {
	vader: data.map(obj => [ obj.sentiment_human, obj.sentiment_vader ]),
	bart: data_noneutral.map(obj => [ obj.sentiment_human, obj.sentiment_bart ]),
	transformer: data_noneutral.map(obj => [ obj.sentiment_human, obj.sentiment_transformer ]),
	clip: data_noneutral.filter(obj => obj.sentiment_clip.length > 0)
		.map(obj => [ obj.sentiment_human, obj.sentiment_clip ])
}

const results = {
	vader: calculate_stats(acc.vader, true),
	bart: calculate_stats(acc.bart),
	transformer: calculate_stats(acc.transformer),
	clip: calculate_stats(acc.clip)
};


switch(mode) {
	case "csv":
		console.log("model\t"+Object.keys(results.transformer).join(`\t`));
		for(const model_name in results) {
			const row = [ model_name, ...Object.values(results[model_name]).map(el => el.toFixed(decimal_places)) ];
			console.log(row.join(`\t`));
		}
		break;
	case "json":
		console.log(JSON.stringify(results, null, "\t"));
		break;
}
