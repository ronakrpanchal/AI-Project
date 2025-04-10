# Comment Filtering System

A machine learning-based project to filter inappropriate, offensive, or toxic comments from user-generated content on social media platforms and e-commerce websites. The system flags harmful comments like abuse, threats, or hate speech, while allowing genuine negative feedback about products or posts.

---

## 📌 Features
- Detects toxic language including threats, hate speech, and profanity.
- Allows critical yet respectful feedback to be posted.
- Trained on a real-world dataset with labeled toxic comment categories.
- Deployed using Streamlit to simulate a social media-style interface.

---

## 🧠 ML Model
- **Vectorizer**: TF-IDF
- **Classifier**: Logistic Regression
- **Trained Labels**: toxic (binary)

---

## 📁 Project Structure
```
AI Project/
├── app/
│   └── app.py                # Streamlit app UI for comment filtering
├── data/
│   └── cleaned_dataset.csv   # Preprocessed dataset
├── models/
│   ├── comment_filter_model.pkl  # Trained ML model
│   └── vectorizer.pkl           # TF-IDF vectorizer
├── notebooks/
│   ├── preprocessing.ipynb     # Step 1 & 2: Data Cleaning
│   └── eda_and_models.ipynb    # Step 3–5: EDA, Training, Evaluation
└── my_image.jpeg               # Sample user image for post
```

---

## 🚀 How to Run
1. Clone the repo.
2. Install requirements:
```bash
pip install -r requirements.txt
```
3. Launch the Streamlit app:
```bash
streamlit run app/app.py
```

---

## 🖼️ Streamlit UI
- Simulates a social media post (image + caption).
- Users can type a comment.
- If clean: it gets posted.
- If flagged: it shows an error message.

---

## ✅ Completed Steps
- ✅ Dataset collection & cleaning
- ✅ Text preprocessing (lemmatization, lowercasing, etc.)
- ✅ EDA & label distribution
- ✅ Model training and evaluation
- ✅ Streamlit deployment

---

## ✨ Future Improvements
- Support for multi-label classification (toxic, threat, obscene, etc.)
- Integration with databases or user login
- Use of LLMs or transformers for better accuracy

---

## 👤 Author
Ronak Panchal

---

## 📃 License
This project is for educational purposes only.

