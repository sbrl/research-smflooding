#!/usr/bin/env node
"use strict";

import fs from 'fs';
import nexline from 'nexline';

const filename_csv = process.argv[2];
const filename_tweets = process.argv[3];
const filename_out = process.argv[4];

const tweets = [];
const csv = [];

///
// 1: Read tweets
///
const reader = nexline({ input: fs.createReadStream(filename_tweets) });
let i_tweets = -1;
while(true) {
	i_tweets++;
	const line = await reader.next();
	if(line == null) break;
	if(line.trim().length == 0) continue;
	
	tweets.push(JSON.parse(line));
	
	if(i_tweets % 1000 === 0) process.stderr.write(`Loaded ${i_tweets} tweets\r`);
}

const reader_csv = nexline({ input: fs.createReadStream(filename_csv) });
let i = -1, header;
while(true) {
	i++;
	const line = await reader_csv.next();
	if(line == null) break;
	if(line.trim().length == 0) continue;
	
	const parts = line.split("\t");
	
	if(i === 0) {
		header = parts;
		continue;
	}
	csv.push(parts);
}


///
// 2: Matching & writing
///
header.push("sentiment_emoji");

const stream_out = fs.createWriteStream(filename_out);
let success = 0, failure = 0;
stream_out.write(header.join("\t") + "\n");
for(const row of csv) {
	const found_tweet = tweets.find(tweet => tweet.id == row[0]);
	
	if(!found_tweet) {
		console.error(`FAILED row`, row, `found_tweet`, found_tweet);
		failure++;
		continue;
	}
	
	success++;
	row.push(found_tweet.label_cats);
	stream_out.write(row.join("\t") + "\n");
}
console.log(`SUCCESS ${success} FAILURE ${failure}`);

await stream_out.close();
