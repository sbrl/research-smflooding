# PhD-Social-Media

Social media AI stuff for analysing the sentiment of floods. Done for my PhD.

See also https://www.npmjs.com/package/twitter-academic-downloader, which used to be called PhD-Social-Media before it turned into a twitter downloader.

**TODO: Fill this README out with comprehensive instructions.**

## Entrypoints
This project has a large number of entrypoints, depending on what you want to do. The reasoning for this is complicated, but a major factor was significant difficulties were encountered when attempting to implement a subcommand-based system.

In the future, a bash-based wrapper to the below scripts will be implemented.

 - `src/text_classifier.py`: The main LSTM (or transformer)-based text classifier
 - `src/image_classifier.py`: Using a sentiment analysis model from text_classifier.py, train a model to classify images associated with tweets.
 - `src/confusion_matrix.py`: Makes a confusion matrix using a given dataset & saved model checkpoint.
 - `src/confusion_matrix_image.py`: Same as `confusion_matrix.py`, but for image classification models see `image_classifier.py`.
 - `src/data_splitter.py`: Splits the specified file of tweets up into multiple separate files based on a given category file. Used when balancing dataset.
 - `src/label_tweets.py`: Labels a JSONL tweets file using the given pre-trained sentiment analysis model (see `text_classifier.py`)
 - `src/label_images.py`: Similar to `label_tweets.py`, but for image (see `image_classifier.py`)
 - `src/label_tweets_topics.py`: Label a JSONL tweets file using a pre-trained LDA model (LSA/LSI models are NOT supported because gensim does not appear to support query trained LSIModel instances). Labels use a different key to src/label_images.py, so can coexist therewith.
 - `src/find_topics.py`: Run an LDA topic analysis over the specified tweets file.
 - `src/glove_longest.py`: Finds the longest sequence of tokens in the input
 - `src/test_glove.py`: Simple test script to check the GloVe parsing & conversion.


## Useful commands

### Tally topics against sentiment
```bash
jq --raw-output '[.label_topic, .label] | @tsv' <./output/lda-all-t20/tweets-all-new-20220117-labelled.jsonl | sort | uniq -c | sort -k2,2n -k3 | paste -s -d' \n' | awk 'BEGIN { OFS="\t"; print("topic", "negative", "positive", "order"); } { print($2, $1, $4, $3 "-" $6); }'
```

Example output:

```
topic	negative	positive	order
0	2182	5354	negative-positive
1	4513	4831	negative-positive
2	9430	1808	negative-positive
3	3812	5767	negative-positive
...
```

### Tally topics against media urls
To join topic ids against *image* sentiment, do this:

```bash
cat path/to/tweets-labelled.jsonl | jq --raw-output -c 'select(has("media")) | [ .label, .label_topic, (.media[] | select(.type=="photo") | .url) ] | @tsv' | awk 'BEGIN{OFS="\t";} {for(i=3; i<NF; i++) { sub(".*/", "", $i); print($i, $1, $2); }}' | csvjoin -c 1 -H -t - path/to/media-labels.tsv | sed -e '1s/.*/filename,sentiment_tweet,topic_id,sentiment_image/' | tr "," "\t" >path/to/output.jsonl
```

This produces a TSV file out that has the following columns:

1.  **`filename`:** The filename of the media file in question
2.  **`sentiment_tweet`:** The sentiment of the _tweet_
3.  **`topic_id`:** The LDA topic ID
4.  **`sentiment_image`:** The sentiment of the _image_

Note that the input file must be labelled by both the tweet classifier (transformer or LSTM) AND an LDA model to use this function. In addition, an image classification model must be also trained and used to label all input images beforehand too (see the `src/image_classifier.py` and `src/label_images.py` endpoints and their associated SLURM scripts).

### Then, to compare media sentiment with topic id:
```bash
cut -f 3-4 <output/lda-all-t20/media_sentiment-tweets_topic_sentiment-media.tsv | tail -n +2 | sort -n | uniq -c | sort -k2,2n -k 3,3 | paste -s -d' \n' | awk 'BEGIN { OFS="\t"; print("topic", "media-negative", "media-positive", "order"); } { print($2, $1, $4, $3 "-" $6); }' >output/lda-all-t20/topics-media-sentiments.tsv
```

The columns this generates as output have pretty much the same meaning as the *Tally topics against sentiment* example above.

### Alternatively, to sample images:
With media filenames annotated with topic ids and associated sentiment, we can now sample images with this bash 3-liner:

```bash
targetdir="/tmp/topicsample";
# Function to sample for a single topic id and sentiment
sample() { mediadir=/mnt/research-data/main/twitter/media; rootdir="$1"; number="$2"; sentiment="$3"; mkdir -p "${rootdir}/${number}/${sentiment}"; tail -n+2 media_sentiment-tweets_topic_sentiment-media.tsv | awk -v "sentiment=${sentiment}" -v "number=${number}" '$3 == number && $4 == sentiment { print $1 }' | shuf | head | xargs -I {} cp "${mediadir}/{}" "${rootdir}/${number}/${sentiment}/{}"; }
# Sample for all topic ids and sentiments
for i in {0..19}; do sample /tmp/topicsample $i positive; sample /tmp/topicsample $i negative; done
# Montage the samples together
find "${targetdir}" -mindepth 2 -maxdepth 2 -type d -print0 | xargs -P "$(nproc)" -0 -I {} sh -c 'dir="{}"; echo "${dir}"; montage $(find "${dir}" -type f) -geometry 512x512+10+10 -tile 10x1 "${dir}.png"';
find "${targetdir}" -name 'positive.png' -print0 | xargs -P "$(nproc)" -0 -n1 mogrify -bordercolor '#43ad29' -border 25x25;
find "${targetdir}" -name 'negative.png' -print0 | xargs -P "$(nproc)" -0 -n1 mogrify -bordercolor '#ad2929' -border 25x25;
find "${targetdir}" -mindepth 2 -maxdepth 2 -type f -print0 | xargs -P "$(nproc)" -0 -I {} sh -c 'file="{}"; convert "${file}" -gravity center -pointsize 100 -size 500x140 label:"$(basename "$(dirname "${file}")") $(basename "${file%.*}")" -append -resize 9999x375 "${file}.png"; mv -f "${file}.png" "${file}";';
convert -append $(find "${targetdir}" -mindepth 2 -maxdepth 2 -type f | sort -t '/' -k 4,4n -k 5,5) /tmp/result.jpeg
```

...replace `/mnt/research-data/main/twitter/media` with the path to the directory containing your downloaded media images, `0` and `19` with the min/max topic ids (`0` and `19` are for 20 topics), and `/tmp/topicsample` with the target directory to sample to.



## Sources and further reading
 - [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer/) [Keras]
 - [The Illustrated Transformer - Representing The Order of The Sequence Using Positional Encoding](https://jalammar.github.io/illustrated-transformer/#representing-the-order-of-the-sequence-using-positional-encoding)
 - LDA Topic modelling
     - [GenSim LdaModel](https://radimrehurek.com/gensim/models/ldamodel.html)
     - [Topic Modelling With LDA - A Hands-on Introduction](https://www.analyticsvidhya.com/blog/2021/07/topic-modelling-with-lda-a-hands-on-introduction/)
     - <https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/>
     - <https://www.tutorialspoint.com/gensim/gensim_creating_lda_topic_model.htm>
