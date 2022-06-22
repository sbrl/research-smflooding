#!/usr/bin/env node
"use strict";


import fs from 'fs';

import { fOneScoreMacro as f1score } from 'data-science-js';

const filepath = process.argv[2];
const key_groundtruth = "sentiment_emoji";
const groundtruth_contains_neutral = false;
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

function calculate_stats(arr) {
	const trans_table = {
		positive: 1,
		neutral: 0,
		negative: -1
	};
	
	// 1: Convert to numbers
	let numerical = arr.map(row => row.map(cell => {
		if(typeof cell !== "string" || typeof trans_table[cell] !== "number") {
			console.error("trans_table failure | ROW", row, "CELL", cell);
			process.exit();
		}
		return trans_table[cell];
	}));
	
	if(!groundtruth_contains_neutral) {
		// rewrite neutral classifications to positive
		for(const row of numerical) {
			if(row[1] == 0) row[1] = 1;
		}
	}
	
	// 2: Do calculation
	const result = f1score(
		numerical.map(row => row[0]),
		numerical.map(row => row[1])
	);
	
	// 3: Add in accuracy
	if(typeof result.accuracy !== "undefined")
		throw new Error(`Accuracy exists when it shouldn't`);
	// if(contains_neutral)
	result.accuracy = arr.filter(row => row[0] === row[1]).length / arr.length;
	// else
	// 	result.accuracy = arr.filter(row => row[0] !== `neutral`)
	// 		.filter(row => row[0] === row[1]).length / arr.length;
	result.samples = numerical.length;
	result.truth_positive = numerical.filter(row => row[0] === 1).length;
	result.predict_positive = numerical.filter(row => row[1] === 1).length;
	result.truth_neutral = numerical.filter(row => row[0] === 0).length;
	result.predict_neutral = numerical.filter(row => row[1] === 0).length;
	result.truth_negative = numerical.filter(row => row[0] === -1).length;
	result.predict_negative = numerical.filter(row => row[1] === -1).length;
	
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
	.filter(obj => obj[key_groundtruth].length > 0)

console.error(`DATA`, data.length);

// const data_noneutral = data.filter(obj => obj[key_groundtruth] !== "neutral");

const acc = {
	vader: data.filter(obj => obj.sentiment_vader.length > 0)
		.map(obj => [ obj[key_groundtruth], obj.sentiment_vader ]),
	bart: data.filter(obj => obj.sentiment_bart.length > 0)
		.map(obj => [ obj[key_groundtruth], obj.sentiment_bart ]),
	transformer: data.filter(obj => obj.sentiment_transformer.length > 0)
		.map(obj => [ obj[key_groundtruth], obj.sentiment_transformer ]),
	lstm: data.filter(obj => obj.sentiment_lstm.length > 0)
		.map(obj => [ obj[key_groundtruth], obj.sentiment_lstm ]),
	// clip: data.filter(obj => obj.sentiment_clip.length > 0)
	// 	.map(obj => [ obj[key_groundtruth], obj.sentiment_clip ]),
	resnet: data.filter(obj => obj.sentiment_resnet.length > 0)
		.map(obj => [ obj[key_groundtruth], obj.sentiment_resnet ])
}

console.error(`COUNTS: vader ${acc.vader.length} bart ${acc.bart.length} transformer ${acc.transformer.length} lstm ${acc.lstm.length} resnet ${acc.resnet.length}`);

const results = {
	vader: calculate_stats(acc.vader),
	bart: calculate_stats(acc.bart),
	transformer: calculate_stats(acc.transformer),
	lstm: calculate_stats(acc.lstm),
	// clip: calculate_stats(acc.clip)
	resnet: calculate_stats(acc.resnet),
};


switch(mode) {
	case "csv":
		console.log("model\t"+Object.keys(results.transformer).join(`\t`));
		for(const model_name in results) {
			const row = [ model_name, ...Object.values(results[model_name]).map(el => el.toFixed(decimal_places).replace(/\.?0+$/, "")) ];
			console.log(row.join(`\t`));
		}
		break;
	case "json":
		console.log(JSON.stringify(results, null, "\t"));
		break;
}
