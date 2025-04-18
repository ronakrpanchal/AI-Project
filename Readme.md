# Comment Filtering System (Hybrid: ML + DL)

An AI-based system that filters toxic or inappropriate comments across social platforms and e-commerce reviews. The system supports both classical machine learning and deep learning approaches. It enhances toxicity detection using spell correction and data augmentation.

---

## ✨ Features

- Detects harmful content: threats, hate speech, obscenity, etc.
- Uses DNN with BiLSTM and tokenizer for better generalization.
- Separate Flask API for model inference.
- Streamlit frontend for traditional models.

---

## 🧠 ML & DL Models

### ML Models:
- **Vectorizer**: TF-IDF
- **Classifiers**: Logistic Regression, Decision Tree, Random Forest

### DNN Model:
- **Tokenizer**: Custom tokenizer (pickled)
- **Architecture**: Embedding → BiLSTM → Dense → Sigmoid
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Precision, Recall, F1-score

---

## 📈 Evaluation

- Multi-label classification (6 toxicity categories)
- Precision, Recall, F1-score computed using `sklearn.metrics`
- Trained with `np.random.seed(42)` and `tf.random.set_seed(42)` for reproducibility

---

## 🔥 NLP Tricks Used

- **Custom Preprocessing**: Lowercasing, punctuation removal, padding sequences

---

## 🧪 Testing Edge Cases

- ✅ Detected `I will kill you` as toxic
- ❌ Bypassed standard models (resolved using DNN + NLP hacks)

---

## 📂 Dataset

- [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)

---

## File Structure

```
project-root/
├── api/
│   ├── api.py
│   ├── comments.db
│   └── templates/
│
├── app/
│   ├── app.py
│   └── my_image_2.png
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── test_labels.csv
│
├── models/
│   ├── decision_tree.pkl
│   ├── dnn_model.h5
│   ├── log_reg.pkl
│   ├── random_forest.pkl
│   ├── Tokenizer.pkl
│   └── vectorizer.pkl
│
├── notebooks/
│   ├── comment-filteration.ipynb
│   ├── eda.ipynb
│   └── models.ipynb
│
├── src/
│   └── preprocess.ipynb
│
└── requirements.txt
```

## 🚀 How to Run

### 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Launch Streamlit App (ML Models)

```bash
streamlit run app/app.py
```

### 🔹 Run Flask API (DNN Model)

```bash
cd api
python main.py
```

---

## ✅ Completed

- ✅ Dataset preprocessing
- ✅ Classical model training (TF-IDF + ML)
- ✅ DNN model building, training, and saving (`.h5`)
- ✅ Spellchecker and text augmentations
- ✅ Flask API for DL model
- ✅ Project folder structure cleanup

---

## 🔮 Future Scope

- Integrate HuggingFace Transformers (BERT)
- Offensive language translation to context-safe summaries
- Integrate with databases & auth
- Deploy Flask & Streamlit on the cloud (Heroku, Render, etc.)

---

## 👤 Author

Ronak Panchal