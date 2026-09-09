# 🎫 Smart Ticket Routing — AI Support Ticket Classifier

> **Turn messy customer complaints into structured, actionable support decisions.**

**Smart Ticket Routing** is an NLP-powered Machine Learning system that automatically analyzes customer support tickets and predicts two critical attributes:

* 🏷️ **Ticket Category**
* 🚨 **Priority Level**

Instead of manually sorting incoming tickets, the system processes raw customer messages, converts them into numerical representations using **TF-IDF**, and passes them through trained classification models to generate instant predictions.

Built with **Python, Scikit-learn, NLTK/spaCy-compatible preprocessing, and Joblib**.

---

## 🚀 How It Works

```text
Customer Support Ticket
          │
          ▼
   Text Preprocessing
          │
          ▼
      TF-IDF
   Vectorization
          │
          ▼
   ┌──────┴──────┐
   ▼             ▼
Category Model  Priority Model
   │             │
   ▼             ▼
Category       Priority
```

### Example

**Input:**

```text
URGENT!! My screen went completely black while using the app
and now it won't load. I am on the premium plan, please fix
this immediately!
```

**Output:**

```text
Predicted Category: Technical Issue
Assigned Priority: High
```

---

## 🧠 Core Features

### 📝 NLP Text Preprocessing

Raw ticket text is cleaned before being passed to the ML pipeline.

The preprocessing stage handles operations such as:

* Text normalization
* Lowercasing
* Removing unnecessary characters
* Tokenization
* Stopword handling
* Preparing text for vectorization

The preprocessing logic is encapsulated inside:

```text
src/text_preprocessor.py
```

---

### 🔢 TF-IDF Feature Extraction

The cleaned ticket is transformed into a numerical representation using a trained **TF-IDF Vectorizer**.

```python
vectorized_text = vectorizer.transform([cleaned_text])
```

This allows the classification models to work with textual data as numerical feature vectors.

The trained vectorizer is stored as:

```text
models/tfidf_vectorizer.pkl
```

---

## 🏷️ Ticket Category Classification

The system uses a trained classification model to determine what type of issue the customer is reporting.

```python
pred_category = cat_model.predict(vectorized_text)[0]
```

The trained category model is loaded from:

```text
models/category_model.pkl
```

Depending on the dataset, categories can represent issues such as:

* Billing
* Technical Issues
* Account Problems
* Product Issues
* General Queries

---

## 🚨 Priority Prediction

The second classification model predicts how urgently the ticket should be handled.

```python
pred_priority = priority_model.predict(vectorized_text)[0]
```

The priority model is stored as:

```text
models/priority_model.pkl
```

Typical priority levels include:

```text
🔴 High
🟡 Medium
🟢 Low
```

This allows support teams to identify urgent issues without manually reviewing every ticket.

---

## ⚙️ Inference Pipeline

The prediction pipeline is intentionally lightweight and reusable.

```python
def predict_ticket(raw_ticket_text):

    cleaned_text = preprocessor.clean_text(raw_ticket_text)

    vectorized_text = vectorizer.transform([cleaned_text])

    pred_category = cat_model.predict(vectorized_text)[0]

    pred_priority = priority_model.predict(vectorized_text)[0]

    return {
        "Original Ticket": raw_ticket_text,
        "Predicted Category": pred_category,
        "Assigned Priority": pred_priority
    }
```

The pipeline separates **preprocessing, feature extraction, and prediction**, making the system easy to extend or integrate into another application.

---

## 🛠️ Tech Stack

| Component           | Technology                  |
| ------------------- | --------------------------- |
| Language            | Python                      |
| Machine Learning    | Scikit-learn                |
| NLP                 | NLTK / Custom preprocessing |
| Feature Extraction  | TF-IDF                      |
| Model Serialization | Joblib                      |
| Development         | VS Code / Jupyter           |
| Version Control     | Git & GitHub                |

---

## 📂 Project Structure

```text
FUTURE_ML_04/
│
├── data/
│   └── support_tickets.csv
│
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── category_model.pkl
│   └── priority_model.pkl
│
├── src/
│   ├── text_preprocessor.py
│   └── train.py
│
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔄 Prediction Flow

The complete system follows this pipeline:

```text
Raw Ticket
    ↓
Clean & Normalize Text
    ↓
TF-IDF Transformation
    ↓
 ┌─────────────────────┐
 │                     │
 ▼                     ▼
Category Classifier   Priority Classifier
 │                     │
 ▼                     ▼
Ticket Category       Priority Level
```

A single ticket is therefore processed by **two independent prediction models**.

---

## 📊 Machine Learning Approach

The project treats ticket routing as a **supervised text classification problem**.

### Input

Unstructured customer support ticket text.

### Features

TF-IDF representation of the cleaned ticket.

### Outputs

Two predictions:

```text
Ticket → Category
Ticket → Priority
```

This approach is lightweight, interpretable, and fast enough for high-volume text classification workloads.

---

## ▶️ Run the Project

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd FUTURE_ML_04
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Train the Models

Before running predictions, make sure the trained model artifacts exist.

Run:

```bash
python src/train.py
```

This should generate:

```text
models/
├── tfidf_vectorizer.pkl
├── category_model.pkl
└── priority_model.pkl
```

---

## 🎯 Make a Prediction

Run:

```bash
python predict.py
```

The inference script processes the sample ticket and returns:

```text
Original Ticket: ...
Predicted Category: ...
Assigned Priority: ...
```

You can also modify the `sample_ticket` variable in `predict.py` to test your own support tickets.

---

## 💼 Business Value

In a real customer-support environment, incoming tickets can quickly become difficult to manage manually.

Smart Ticket Routing can help automate the first stage of the support workflow:

```text
Incoming Ticket
      ↓
Automatic Classification
      ↓
Priority Assignment
      ↓
Route to Appropriate Team
      ↓
Faster Response
```

Potential benefits include:

* ⚡ Faster ticket triage
* 📉 Reduced manual workload
* 🚨 Faster identification of urgent issues
* 🏷️ Consistent ticket categorization
* 📊 Better support-team organization
* 🔄 Scalable processing of large ticket volumes

---

## 🔮 Future Improvements

The current system provides a strong classical NLP baseline. Possible upgrades include:

* 🤗 Transformer-based text embeddings
* 🧠 BERT / DistilBERT classification
* 🔍 Confidence scores for predictions
* 📊 Interactive Streamlit dashboard
* 🔄 Automatic ticket routing to teams
* 📧 Email/helpdesk integration
* 📈 Real-time support analytics
* 🗃️ Database-backed ticket management
* ⚡ Batch prediction for large ticket pipelines

---

## 🎓 What This Project Demonstrates

This project demonstrates practical experience with:

* Natural Language Processing
* Text preprocessing
* TF-IDF feature engineering
* Supervised classification
* Multi-output decision pipelines
* Model serialization and reuse
* Production-style inference
* Business-oriented Machine Learning

---

## 👨‍💻 Author

**Praneeth Varma Uddaraju**

Machine Learning • NLP • Python

Built as part of the **Future Interns Machine Learning Internship — Support Ticket Classification & Prioritization Task**.

---

## ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐.
