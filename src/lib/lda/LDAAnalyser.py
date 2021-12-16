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
            id2word=
            num_topics = self.count_topics
        )
    
    def save(filepath):
        self.model.save(filepath)
    
    def load(filepath):
        self.model = LdaModel.load(filepath)
    
    
