import os
import sys
import joblib

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

from src.text_preprocessor import TicketPreprocessor

def predict_ticket(raw_ticket_text):
    # 1. Initialize preprocessor and clean the text input
    preprocessor = TicketPreprocessor()
    cleaned_text = preprocessor.clean_text(raw_ticket_text)
    
    # 2. Paths to saved artifacts
    vectorizer_path = os.path.join(root_dir, 'models', 'tfidf_vectorizer.pkl')
    cat_model_path = os.path.join(root_dir, 'models', 'category_model.pkl')
    priority_model_path = os.path.join(root_dir, 'models', 'priority_model.pkl')
    
    # Check if models exist
    if not (os.path.exists(vectorizer_path) and os.path.exists(cat_model_path) and os.path.exists(priority_model_path)):
        return "❌ Error: Trained models not found. Please run 'python src/train.py' first to train the system."
    
    # 3. Load the saved ML artifacts
    vectorizer = joblib.load(vectorizer_path)
    cat_model = joblib.load(cat_model_path)
    priority_model = joblib.load(priority_model_path)
    
    # 4. Transform text using the loaded vectorizer
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # 5. Generate predictions
    pred_category = cat_model.predict(vectorized_text)[0]
    pred_priority = priority_model.predict(vectorized_text)[0]
    
    return {
        "Original Ticket": raw_ticket_text,
        "Predicted Category": pred_category,
        "Assigned Priority": pred_priority
    }

if __name__ == "__main__":
    print("\n🎫 --- Smart Ticket Routing Pipeline --- 🎫\n")
    
    # Sample unseen customer issue
    sample_ticket = "URGENT!! My screen went completely black while using the app and now it won't load. I am on the premium plan, please fix this immediately!"
    
    result = predict_ticket(sample_ticket)
    
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"**{key}:** {value}")
    else:
        print(result)