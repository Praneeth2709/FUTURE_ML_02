import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TicketPreprocessor:
    def __init__(self):
        # Safely download required NLTK resources if not present
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
            nltk.download('punkt_tab')  # Added this line
            self.stop_words = set(stopwords.words('english'))
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # 1. Convert text to lowercase
        text = text.lower()
        
        # 2. Remove punctuation, special characters, and numbers
        text = re.sub(re.compile(r'[^a-zA-Z\s]'), '', text)
        
        # 3. Tokenize (break sentences down into individual words)
        tokens = word_tokenize(text)
        
        # 4. Remove stop words (e.g., "is", "the", "at" which don't carry ML value)
        cleaned_tokens = [word for word in tokens if word not in self.stop_words]
        
        # 5. Re-join the cleaned words back into a single string
        return " ".join(cleaned_tokens)