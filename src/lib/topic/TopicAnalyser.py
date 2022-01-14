import re

from pprint import pprint
import pysnooper
from loguru import logger

from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
from gensim.models.ldamodel import LdaModel
# Apparently the auto settings we're using don't work with LDAmulticore :-/
# from gensim.models.ldamulticore import LdaMulticore
from gensim.models.lsimodel import LsiModel
from gensim.corpora.dictionary import Dictionary

class TopicAnalyser:
    def __init__(self, count_topics = 10, words_per_topic = 10):
        self.count_topics = count_topics
        self.words_per_topic = words_per_topic
    
    
    # Convenience function to reduce repetition
    def re_sub(self, pattern, repl, text):
    	return re.sub(pattern, repl, text, flags=re.MULTILINE | re.DOTALL)
    
    def preprocess_str(self, string):
        string = self.re_sub(r"@\S+", " ", string)
        string = self.re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " ", string)
        return preprocess_string(string)
    
    
    def preprocess_dataset(self, dataset):
        return [ self.preprocess_str(string) for string in dataset ]
    
    def train(self, model, dataset):
        
        logger.info("Preprocessing data [1 / 5]")
        self.dataset = self.preprocess_dataset(dataset)
        
        logger.info("Compiling dictionary [2 / 5]")
        self.dictionary = Dictionary(self.dataset)
        logger.info("Vectorising data [3 / 5]")
        self.dataset_vec = [ self.dictionary.doc2bow(item) for item in self.dataset ]
        
        
        if model == "lda":
            return self.do_lda(self.dataset)
        elif model == "lsa":
            return self.do_lsa(self.dataset)
        else:
            raise Exception(f"Error: Unknown model type '{model}' (valid types: lda [default], lsa).")
    
    def do_lsa(self, dataset):
        logger.info("Constructing LDA [4 / 5]")
        self.model = LsiModel(
            corpus=self.dataset_vec,
            num_topics=self.count_topics,
            id2word=self.dictionary,
            distributed=False, # Apparently uses a complicated networking setup
        )
        
        logger.info("Calculating statistics")
        topics = self.model.print_topics(
            num_topics=self.count_topics,
            num_words=self.words_per_topic
        )
        avg_topic_coherence = sum([t[1] for t in topics]) / self.count_topics
        
        return avg_topic_coherence, None, topics
        
    
    def do_lda(self, dataset):
        
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
            texts=self.dataset,
            dictionary=self.dictionary,
            topn=self.words_per_topic
        )
        avg_topic_coherence = sum([t[1] for t in topics]) / self.count_topics
        perplexity = self.model.log_perplexity(self.dataset_vec)
        
        return avg_topic_coherence, perplexity, topics
    
    def save(self, filepath):
        self.model.save(filepath)
    
    def load(self, filepath):
        self.model = LdaModel.load(filepath)
    
