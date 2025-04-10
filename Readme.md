# Comment Filtering System

A machine learning-based project to filter inappropriate, offensive, or toxic comments from user-generated content on social media platforms and e-commerce websites. The system flags harmful comments like abuse, threats, or hate speech, while allowing genuine negative feedback about products or posts.

---

## 📌 Features
- Detects toxic language including threats, hate speech, and profanity.
- Allows critical yet respectful feedback to be posted.
- Trained on a real-world dataset with labeled toxic comment categories.
- Deployed using Streamlit to simulate a social media-style interface.

---

## 📂 Dataset

This project uses the [Jigsaw Toxic Comment Classification Challenge dataset](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data) from Kaggle.

To download:
1. Visit the [dataset page](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)
2. Accept the competition rules
3. Download `train.csv` , `test.csv` and `test_labels.csv` into the `data/` folder

## 🧠 ML Model
- **Vectorizer**: TF-IDF
- **Classifier**: Logistic Regression , Decision Tree , Random Forest
- **Trained Labels**: toxic, severe_toxic, obscene, threat, insult, identity_hate

---

## 📁 Project Structure
```
AI Project/
├── app/
│   ├── app.py                  # Streamlit app for traditional models
├── data/
│   ├── test.csv                # Raw test data
│   ├── test_labels.csv         # Test labels (for evaluation)
│   └── train.csv               # Raw training data
├── models/
│   ├── decision_tree.pkl       # Decision Tree model
│   ├── log_reg.pkl             # Logistic Regression model
│   ├── random_forest.pkl       # Random Forest model
│   └── vectorizer.pkl          # TF-IDF vectorizer
├── notebooks/
│   ├── eda.ipynb               # Exploratory Data Analysis
│   └── models.ipynb            # Training and evaluation of models
├── src/
│   ├── preprocess.ipynb        # Preprocessing logic
│   ├── preprocessed_train.csv  # Cleaned train data
│   └── preprocessed_test.csv   # Cleaned test data
├── Readme.md                   # Project documentation
└── requirements.txt            # Python dependencies
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

