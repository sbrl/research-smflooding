import re

from pprint import pprint
import pysnooper
from loguru import logger

from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

class LDAAnalyser:
    def __init__(self, count_topics = 10):
        self.count_topics = count_topics
    
    
    # Convenience function to reduce repetition
    def re_sub(self, pattern, repl, text):
    	return re.sub(pattern, repl, text, flags=re.MULTILINE | re.DOTALL)
    
    def preprocess_dataset(self, dataset):
        return [ preprocess_string(
            self.re_sub(r"@\w+", " ", string) # this strips all digits, but we may not want it to do that
        ) for string in dataset ]
    
    def train(self, dataset):
        
        logger.info("Preprocessing data [1 / 5]")
        dataset = self.preprocess_dataset(dataset)
        
        logger.info("Compiling dictionary [2 / 5]")
        self.dictionary = Dictionary(dataset)
        logger.info("Vectorising data [3 / 5]")
        self.dataset_vec = [ self.dictionary.doc2bow(item) for item in dataset ]
        
        logger.info("Constructing LDA [4 / 5]")
        self.model = LdaModel(
            self.dataset_vec,
            id2word=self.dictionary,
            num_topics=self.count_topics,
            alpha='auto',
            eta='auto'
        )
        
        logger.info("Calculating statistics [5 / 5]")
        
        # Ref https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html
        topics = self.model.top_topics(
            corpus=self.dataset_vec,
            texts=dataset,
            dictionary=self.dictionary,
            topn=self.count_topics
        )
        avg_topic_coherence = sum([t[1] for t in topics]) / self.count_topics
        perplexity = self.model.log_perplexity(self.dataset_vec)
        
        return avg_topic_coherence, perplexity, topics
    
    
    def save(self, filepath):
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model = LdaModel.load(filepath)
    
