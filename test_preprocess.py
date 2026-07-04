from src.text_preprocessor import TicketPreprocessor
from src.feature_engineering import FeaturePipeline

# 1. Initialize our tools
tp = TicketPreprocessor()
pipeline = FeaturePipeline(max_features=10) # Using a small number just for visualization

# 2. Mock a tiny dataset of 2 clean tickets
tickets = [
    "hello experiencing horrific error login page please help asap",
    "billing issue payment declined double charged refund money"
]

# 3. Transform the text into numerical vectors
tfidf_matrix = pipeline.fit_transform(tickets)

print("\n--- Feature Engineering (TF-IDF) Test ---")
print("Represented numerical matrix shape:", tfidf_matrix.shape) 
print("Feature Names (The vocabulary matrix created):")
print(pipeline.vectorizer.get_feature_names_out())
print("\nFirst ticket as a sparse numerical array:\n", tfidf_matrix.toarray()[0])