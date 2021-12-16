# PhD-Social-Media

Social media AI stuff for my PhD.

See also https://www.npmjs.com/package/twitter-academic-downloader, which used to be called PhD-Social-Media before it turned into a twitter downloader.

**TODO: Fill this README out with comprehensive instructions.**

## Entrypoints

 - `src/text_classifier.py`: The main LSTM-based text classifier
 - `src/confusion_matrix.py`: Makes a confusion matrix using a given dataset & saved model checkpoint.
 - `src/data_splitter.py`: Splits the specified file of tweets up into multiple separate files based on a given category file. Used when balancing dataset.
 - `src/glove_longest.py`: Finds the longest sequence of tokens in the input
 - `src/test_glove.py`: Simple test script to check the GloVe parsing & conversion.


## Sources and further reading
 - [Text classification with Transformer](https://keras.io/examples/nlp/text_classification_with_transformer/) [Keras]
 - [The Illustrated Transformer - Representing The Order of The Sequence Using Positional Encoding](https://jalammar.github.io/illustrated-transformer/#representing-the-order-of-the-sequence-using-positional-encoding)
 - LDA Topic modelling
     - [GenSim LdaModel](https://radimrehurek.com/gensim/models/ldamodel.html)
     - [Topic Modelling With LDA - A Hands-on Introduction](https://www.analyticsvidhya.com/blog/2021/07/topic-modelling-with-lda-a-hands-on-introduction/)
     - <https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/>
     - <https://www.tutorialspoint.com/gensim/gensim_creating_lda_topic_model.htm>
