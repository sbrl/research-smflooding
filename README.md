# research-smflooding

> Some of the code behind the paper "Real-time social media sentiment analysis for rapid impact assessment of floods" [working title]


Social media artificial intelligence for analysing the sentiment of floods. Done for my PhD.

This is a work in progress. The associated paper has not been published yet.

This is not all of the code. Code for other parts of the project can be found here:

 - [sbrl/twitter-academic-downloader](https://github.com/sbrl/twitter-academic-downloader): The command line program written to download the tweets from Twitter, using Twitter's Academic API. This used to be this code base, before it turned into a twitter downloader. It's recommended to install and use this [via npm](https://www.npmjs.com/package/twitter-academic-downloader).
 - [jakegodsall/twitter-floods](https://github.com/jakegodsall/twitter-floods): The code written to geolocate and plot the sentiment of tweets on a map.
 - **This repository:** The main codebase written to train and interact with the AI models tested in this paper. Contains code for the following:
     - LSTM [Tensorflow]
     - Transformer encoder [Tensorflow]
     - Model based on CLIP (multimodal text + image model) [PyTorch]
     - VADER (`AMT_VADER.ipynb`)
     - RoBERTa (`RoBERTa and BERT_Model.ipynb`)


**WARNING:** This codebase is a WORK IN PROGRESS and has **not been peer-reviewed yet**. I am currently submitting the associated journal article for review (I'll insert a link here if and when it gets published).


**TODO: Fill this README out with comprehensive instructions.**

## System Requirements
 - Linux (Windows may work, but is untested)
 - [git](https://git-scm.com/) (`sudo apt install git`)
 - Python 3.8+ (for Tensorflow models etc) and `pip` (`sudo apt install python3-pip`)
 - A GPU is recommended for training models:
     - Preferably 12GB VRAM or more, may work with less but is untested
     - A card supported by both Tensorflow and PyTorch (currently this is just Nvidia cards. PyTorch requires [CUDA compute capability](https://developer.nvidia.com/cuda-gpus) 3.7+; Tensorflow is 3.5+)
 - PyTorch: <https://pytorch.org/get-started/locally/> (select Package → Pip)
 - Various other packages, but these are installed below
 - Say ~50GiB+ disk space if you really dig into this repo a lot
 - Optionally, for analysing data:
     - A RECENT version of Node.js (only for scripts in the [`scripts` directory](scripts/), so omit if you don't want to use them)
     - [`jq`](https://stedolan.github.io/jq/)
 - Some experience using the Linux terminal

## Getting Started
This getting started tutorial assumes you are using Linux, as this is what it was developed on due in part to using the [Viper HPC at the University of Hull](https://hpc.wordpress.hull.ac.uk/).

It is recommended you install and configure [`twitter-academic-downloader`](https://github.com/sbrl/twitter-academic-downloader) **before** using the programs in this repository. All of the programs in this repository assume the data is in the format that `twitter-academic-downloader` generates.

First, clone this git repository:

```bash
git clone https://github.com/sbrl/research-smflooding.git
cd research-smflooding
```

Then, install the Python dependencies:

```bash
pip3 install --user -r requirements.txt
```

Note that you must install PyTorch as per the system requirements **before** running the above command, because how you install PyTorch is dependent on your system.

You are now setup to use the programs in this repository. The programs this this repository are split into multiple entrypoints. These are described below in the [entrypoints section](#entrypoints).

The usage instructions for each one is shown in the command-line help. To access the command-line help for an entrypoint:

```bash
src/MY_ENTRYPOINT_NAME.py --help
```

...where `MY_ENTRYPOINT_NAME.py` is the filename of the entrypoint you want to use.


## Entrypoints
This project has a large number of entrypoints, depending on what you want to do. The reasoning for this is complicated, but a major factor was significant difficulties were encountered when attempting to implement a subcommand-based system.

In the future, a bash-based wrapper to the below scripts will be implemented.

 - `src/text_classifier.py`: The main LSTM (or transformer)-based text classifier. **Start here.**
 - `src/image_classifier.py`: Using a sentiment analysis model from text_classifier.py, train a ResNet50 model to classify images associated with tweets. **This does not work very well.** I was going to talk about it in the journal article, but ultimately cut it due to the word limit.
 - `src/confusion_matrix.py`: Makes a confusion matrix using a given dataset & saved model checkpoint.
 - `src/confusion_matrix_image.py`: Same as `confusion_matrix.py`, but for image classification models see `image_classifier.py`.
 - `src/data_splitter.py`: Splits the specified file of tweets up into multiple separate files based on a given category file. Used when balancing a dataset.
 - `src/label_tweets.py`: Labels a JSONL tweets file using the given pre-trained sentiment analysis model (see `text_classifier.py`). **Use this to make predictions.**
 - `src/label_images.py`: Similar to `label_tweets.py`, but for image (see `image_classifier.py`)
 - `src/label_tweets_topics.py`: Label a JSONL tweets file using a pre-trained LDA model (LSA/LSI models are NOT supported because gensim does not appear to support query trained LSIModel instances). Labels use a different key to `src/label_images.py`, so can coexist therewith. **Use `src/find_topics.py` to train an LDA topic analysis model.**
 - `src/find_topics.py`: Run an LDA topic analysis over the specified tweets file. Originally was going to be included in the paper, but got cut 'cause of the word limit.
 - `src/glove_longest.py`: Finds the longest sequence of tokens in the input
 - `src/test_glove.py`: Simple test script to check the GloVe parsing & conversion.


## Useful commands
I used a lot of Bash one-liners to analyse the data while implementing the programs in this repository. They are documented below for ease of reference.

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


## FAQ

### Can I have a copy of the data / model checkpoints / graphs / other research outputs please?
Absolutely! Please get in touch by sending me an email.

Unfortunately, I am unable to share the Twitter data though due to Twitter's terms and conditions (I anonymised the data, but Twitter only allows me to share the tweet IDs...). However, I *can* share the date ranges and searches I made to obtain the data, allowing you to redownload the data with my [`twitter-academic-downloader`](https://www.npmjs.com/package/twitter-academic-downloader) program which I implemented for this project.

### Why GPL-3.0, and not MIT/Apache2?
The code, models, and other algorithms in this repository took me a huge amount of effort to develop. To this end, I want to ensure:

1. That the work here benefits everyone
2. Future modifications to this work (either by me or others) benefits everyone
3. Transparency

### I'm confused / have some other question
Please [open an issue](https://github.com/sbrl/research-smflooding/issues). If it's private / confidential, please send me an email.


## Contributing
Contributions are very welcome - both issues and pull requests! Please mention in your pull request that you release your work under the GPL-3.0 (see below).

You are likely to encounter bugs in this code.


## Licence
All the code in this repository is released under the GNU Affero General Public License unless otherwise specified. The full license text is included in the [`LICENSE.md` file](./LICENSE.md) in this repository. GNU [have a great summary of the licence](https://www.gnu.org/licenses/#AGPL) which I strongly recommend reading before using this software.

Other licences can *potentially* be negotiated on request.
