from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

class LDAAnalyser:
    def __init__(self, count_topics = 10):
        self.count_topics = 10
    
    def preprocess_dataset(self, dataset):
        return [ preprocess_string(
            remove_stopwords(string)
        ) for string in dataset ]
    
    def train(self, dataset):
        dataset = self.preprocess_dataset(dataset)
        
        self.dictionary = Dictionary(dataset)
        self.dataset_vec = [ self.dictionary.doc2bow(item) for item in dataset ]
        
        self.model = LdaModel(
            dataset_vec,
            id2word=self.dictionary
            num_topics=self.count_topics,
            alpha='auto',
            eta='auto'
        )
        
        # Ref https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        
        topics = model.top_topics(dataset)
        
        return avg_topic_coherence, topics
    
    
    def save(filepath):
        self.model.save(filepath)
    
    def load(filepath):
        self.model = LdaModel.load(filepath)
    
