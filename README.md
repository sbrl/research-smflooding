# PhD-Social-Media

Social media AI stuff for my PhD.

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


## Sources and further reading
 - [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer/) [Keras]
 - [The Illustrated Transformer - Representing The Order of The Sequence Using Positional Encoding](https://jalammar.github.io/illustrated-transformer/#representing-the-order-of-the-sequence-using-positional-encoding)
 - LDA Topic modelling
     - [GenSim LdaModel](https://radimrehurek.com/gensim/models/ldamodel.html)
     - [Topic Modelling With LDA - A Hands-on Introduction](https://www.analyticsvidhya.com/blog/2021/07/topic-modelling-with-lda-a-hands-on-introduction/)
     - <https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/>
     - <https://www.tutorialspoint.com/gensim/gensim_creating_lda_topic_model.htm>
