#!/usr/bin/env node
"use strict";


import fs from 'fs';

import { fOneScoreMacro as f1score } from 'data-science-js';

const filepath = process.argv[2];
const key_groundtruth = "label_emoji";
const groundtruth_contains_neutral = false;
const mode = process.env.OUTPUT_MODE ? process.env.OUTPUT_MODE : "csv";
const decimal_places = process.env.DECIMAL_PLACES ? parseInt(process.env.DECIMAL_PLACES) : 3;
const collapse_neutral_positive = true;

console.error(`Collapse neutral/positive is ${collapse_neutral_positive?"ON":"OFF"}`);

function normalise_sentiment(sentiment) {
	let result = sentiment.toLowerCase().replace(/slightly\s+/, "");
	switch(result) {
		case "neu":
		case "0":
			result = "neutral";
			break;
		case "pos":
		case "1":
			result = "positive";
			break;
		case "neg":
		case "-1":
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
		const tmp = normalise_sentiment(cell);
		if(typeof cell !== "string" || typeof trans_table[tmp] !== "number") {
			console.error("trans_table failure | ROW", row, "CELL", cell, "TMP", tmp);
			console.trace();
			process.exit();
		}
		return trans_table[tmp];
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
	result.accuracy = numerical.filter(row => row[0] === row[1]).length / numerical.length;
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
	.map(JSON.parse)
	.filter(t => typeof t.label_emoji === "string" && t.label_emoji.length > 0);

const acc = {
	vader: data.filter(obj => typeof obj.label_vader == "string" && obj.label_vader.length > 0)
		.map(obj => [ obj[key_groundtruth], obj.label_vader ]),
	roberta: data.filter(obj => typeof obj.label_bart == "string" && obj.label_bart.length > 0)
		.map(obj => [obj[key_groundtruth], obj.label_bart ]),
	transformer: data.filter(obj => typeof obj.label_transformer == "string" && obj.label_transformer.length > 0)
		.map(obj => [ obj[key_groundtruth], obj.label_transformer ]),
	lstm: data.filter(obj => typeof obj.label_lstm == "string" && obj.label_lstm.length > 0)
		.map(obj => [obj[key_groundtruth], obj.label_lstm]),
	clipNA: data.filter(obj => typeof obj.label_clip_augment == "string" && obj.label_clip_augment.length > 0)
		.map(obj => [obj[key_groundtruth], obj.label_clip_augment]),
	clipA: data.filter(obj => typeof obj.label_clip_noaugment == "string" && obj.label_clip_noaugment.length > 0)
		.map(obj => [obj[key_groundtruth], obj.label_clip_augment]),
	resnet: data.filter(obj => typeof obj.label_resnet == "string" && obj.label_resnet.length > 0)
		.map(obj => [ obj[key_groundtruth], obj.label_resnet ])
}

console.error(`COUNTS: vader ${acc.vader.length} roberta ${acc.roberta.length} transformer ${acc.transformer.length} lstm ${acc.lstm.length} resnet ${acc.resnet.length}`);

const results = {
	vader: calculate_stats(acc.vader),
	roberta: calculate_stats(acc.roberta),
	transformer: calculate_stats(acc.transformer),
	lstm: calculate_stats(acc.lstm),
	clipA: calculate_stats(acc.clipA),
	clipNA: calculate_stats(acc.clipNA),
	resnet: calculate_stats(acc.resnet)
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
