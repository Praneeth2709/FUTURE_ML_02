import os
import sys

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(current_dir)

# Add the root directory to Python's path
sys.path.append(root_dir)
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Now these imports will work seamlessly!
from src.text_preprocessor import TicketPreprocessor
from src.feature_engineering import FeaturePipeline

def main():
    dataset_path = os.path.join(root_dir, 'data', 'raw', 'customer_support_tickets.csv')
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset file not found at '{dataset_path}'.")
        print("Please ensure your extracted Kaggle CSV is placed inside data/raw/ and named customer_support_tickets.csv")
        return
        
    print("🚀 Loading dataset...")
    df = pd.read_csv(dataset_path)
    
    # Mapping columns precisely to the Kaggle dataset
    text_column = 'Ticket Description' 
    category_column = 'Ticket Type'      
    priority_column = 'Ticket Priority'
    
    print("🧹 Cleaning raw text descriptions (this might take a minute)...")
    preprocessor = TicketPreprocessor()
    df['cleaned_text'] = df[text_column].apply(preprocessor.clean_text)
    
    print("📊 Vectorizing text data into TF-IDF numerical features...")
    pipeline = FeaturePipeline(max_features=5000)
    X = pipeline.fit_transform(df['cleaned_text'])
    
    y_category = df[category_column]
    y_priority = df[priority_column]
    
    # --- TRAIN CATEGORY MODEL ---
    print("\n🖥️ Training Category Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_category, test_size=0.2, random_state=42)
    
    cat_model = LogisticRegression(max_iter=1000)
    cat_model.fit(X_train, y_train)
    
    print("\n📋 Category Classification Performance:")
    cat_preds = cat_model.predict(X_test)
    print(classification_report(y_test, cat_preds))
    
    # --- TRAIN PRIORITY MODEL ---
    print("\n🖥️ Training Priority Classifier...")
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_priority, test_size=0.2, random_state=42)
    
    priority_model = LogisticRegression(max_iter=1000)
    priority_model.fit(X_train_p, y_train_p)
    
    print("\n📋 Priority Classification Performance:")
    priority_preds = priority_model.predict(X_test_p)
    print(classification_report(y_test_p, priority_preds))
    
    # --- SAVE ARTIFACTS ---
    print("\n💾 Saving models and pipelines to disk...")
    models_dir = os.path.join(root_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    pipeline.save_vectorizer(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    joblib.dump(cat_model, os.path.join(models_dir, 'category_model.pkl'))
    joblib.dump(priority_model, os.path.join(models_dir, 'priority_model.pkl'))
    print("🎉 System training complete! Artifacts safely stored in the /models directory.")

if __name__ == "__main__":
    main()