import pandas as pd
from DataPreProcessing import SpacyEngine, PytorchBERT, GensimEngine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class FeatureExtraction:
    def __init__(self):
        self.spacy = SpacyEngine()
        self.pytorch = PytorchBERT()
        self.gensim = GensimEngine()
        self.dataset = None
    
    def load_dataset(self, dataset_path):
        with open(dataset_path, 'r') as f:
            self.dataset = f.readlines()
            
    def extract_spacy_features(self):
        spacy_features = []
        for text in self.dataset:
            doc = self.spacy.nlp(text)
            spacy_features.append([ent.text for ent in doc.ents])
        return spacy_features

    def extract_pytorch_features(self):
        pytorch_features = []
        for text in self.dataset:
            pytorch_features.append(self.pytorch.predict(text))
        return pytorch_features

    
    def extract_gensim_features(self):
        gensim_features = []
        for text in self.dataset:
            gensim_features.append(self.gensim.process_text(text))
        return gensim_features
    
    def bag_of_words(self):
        # Create a CountVectorizer object
        vectorizer = CountVectorizer()
        
        # Fit the vectorizer on the dataset
        X = vectorizer.fit_transform(self.dataset)
        
        # Get the feature names
        feature_names = vectorizer.get_feature_names()
        
        return X, feature_names
    
    def n_grams(self, n=2):
        # Create a CountVectorizer object with n-grams
        vectorizer = CountVectorizer(ngram_range=(n,n))
        
        # Fit the vectorizer on the dataset
        X = vectorizer.fit_transform(self.dataset)
        
        # Get the feature names
        feature_names = vectorizer.get_feature_names()
        
        return X, feature_names
    
    def tf_idf(self):
        # Create a TfidfVectorizer object
        vectorizer = TfidfVectorizer()
        
        # Fit the vectorizer on the dataset
        X = vectorizer.fit_transform(self.dataset)
        
        # Get the feature names
        feature_names = vectorizer.get_feature_names()
        
        return X, feature_names
    
    def lda(self, n_topics=5):
        # Create a LatentDirichletAllocation object
        lda = LatentDirichletAllocation(n_components=n_topics)
        
        # Fit the LDA on the dataset
        X = lda.fit_transform(self.dataset)
        
        return X
