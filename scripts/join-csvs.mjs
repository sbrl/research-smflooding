#!/usr/bin/env node
"use strict";

import fs from 'fs';

import similarity from 'string-similarity';

const MINIMUM_SIMILARITY = 0.9;	// 90%


function fix_escapes(text) {
	return text.replace(/^[\\"]|[\\"]$/, "")
		.replace(/\\n/g, "\n")
		.replace(/\\r/g, "")
		.replace(/\\t/g, "\t")
		.replace(/\\/g, "")
		.replace(/"/g, "")
		.replace(/\\/g, "");
}

const amt = fs.readFileSync("human-amt.json", "utf8")
	.split("\n")
	.filter(line => line.length > 0)
	.map(line => {
		const parsed = JSON.parse(line);
		return {
			text: parsed["Input.text"],
			sentiment_human: parsed["Answer.sentiment.label"],
		}
	});

const source = fs.readFileSync("NSWFloods-text-withid.json", "utf8")
	.split("\n")
	.filter(line => line.length > 0)
	.map(line => JSON.parse(line));
	
const vader = fs.readFileSync("flood_amt_labelled2-vader.json", "utf8")
	.split("\n")
	.filter(line => line.length > 0)
	.map(line => JSON.parse(line));

const bart = fs.readFileSync("NSWFloods-bart.jsonl", "utf8")
	.split("\n")
	.filter(line => line.length > 0)
	.map(line => JSON.parse(line));

//NSWFloods-bart.csv

const source_as_text = source.map(obj => obj.text);
	
for(const obj of amt) {
	obj.text = fix_escapes(obj.text).trim();
}
for(const obj of source)
	obj.text = fix_escapes(obj.text).trim();

const fixed = [];
const failed = [];

let i = 0;
for(const obj_amt of amt) {
	process.stderr.write(`Processing tweet ${i}\r`);
	
	const sentiment_vader = vader[i].sent_score;
	let sentiment_bart = bart[i].zsc_sentiment;
	if(sentiment_bart < 0) sentiment_bart = "negative";
	if(sentiment_bart > 0) sentiment_bart = "positive";
	if(sentiment_bart === 0) sentiment_bart = "neutral";
	
	let found = false;
	for(const obj_source of source) {
		if(obj_amt.text === obj_source.text || obj_source.text.startsWith(`${obj_amt.text},`)) {
			fixed.push({
				id: obj_source.id,
				text: obj_amt.text,
				sentiment_human: obj_amt.sentiment_human,
				sentiment_vader,
				sentiment_bart
			});
			found = true;
			break;
		}
	}
	if(found) continue;
	
	const candidates = similarity.findBestMatch(obj_amt.text, source_as_text);
	
	if(candidates.bestMatch.rating > MINIMUM_SIMILARITY) {
		fixed.push({
			id: source[candidates.bestMatchIndex].id,
			text: obj_amt.text,
			sentiment_human: obj_amt.sentiment_human,
			sentiment_vader,
			sentiment_bart
		});
		continue;
	}
	
	failed.push(obj_amt);
	i++;
}

fs.writeFileSync(`results-${new Date().toISOString()}.json`, JSON.stringify({
	fixed,
	failed
}));
console.log("FIXED", fixed.length, fixed.slice(0, 10));
console.log("FAILED", failed.length, failed.slice(0, 10));
