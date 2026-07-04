import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

class FeaturePipeline:
    def __init__(self, max_features=5000):
        # max_features keeps the top 5000 most frequent words to prevent the dataset from getting too massive
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        
    def fit_transform(self, text_series):
        """Trains the vectorizer on the text data and transforms it into numerical features."""
        return self.vectorizer.fit_transform(text_series)
        
    def transform(self, text_series):
        """Transforms new unseen text data using the already trained vectorizer."""
        return self.vectorizer.transform(text_series)
        
    def save_vectorizer(self, filepath):
        """Saves the trained vectorizer so we can use it later for predictions."""
        joblib.dump(self.vectorizer, filepath)
        
    def load_vectorizer(self, filepath):
        """Loads a saved vectorizer from disk."""
        self.vectorizer = joblib.load(filepath)