# PhD-Social-Media

Social media AI stuff for my PhD.

See also https://www.npmjs.com/package/twitter-academic-downloader, which used to be called PhD-Social-Media before it turned into a twitter downloader.


## Entry points

 - `src/lstm_text_classifier.py`: The main LSTM-based text classifier
 - `src/data_splitter.py`: Splits the specified file of tweets up into multiple separate files based on a given category file. Used when balancing dataset.
 - `src/glove_longest.py`: Finds the longest sequence of tokens in the input
 - `src/test_glove.py`: Simple test script to check the GloVe parsing & conversion.
