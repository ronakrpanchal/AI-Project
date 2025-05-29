# flask_social_comment_app.py

from flask import Flask, request, render_template, redirect, url_for
import joblib
import sqlite3
import os
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load environment variables
load_dotenv()

# Define toxic labels
TOXIC_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_LEN = 200

# VECTORIZER = os.getenv("VECTORIZER_PATH")
# MODEL = os.getenv("LOG_REG_PATH")

TOKENIZER = os.getenv("TOKENIZER_PATH")
MODEL = os.getenv("DNN_MODEL_PATH")

app = Flask(__name__)

# Load model and vectorizer
# model = joblib.load(MODEL)
# vectorizer = joblib.load(VECTORIZER)
tokenizer = joblib.load(TOKENIZER)
model = load_model(MODEL)

# Initialize database
DB_FILE = "comments.db"

def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                comment TEXT NOT NULL,
                flagged TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

@app.route('/', methods=['GET', 'POST'])
def home():
    error_message = None
    success_message = None

    if request.method == 'POST':
        comment = request.form['comment']
        username = "demo_user"  # In a real app, this comes from login session

        # vec = vectorizer.transform([comment])
        # pred = model.predict(vec)[0]

        # toxic_labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        # flagged_labels = [label for label, p in zip(toxic_labels, pred) if p == 1]
        
        # comment = normalize_and_correct(comment)
        
        # Tokenize and pad the input comment
        seq = tokenizer.texts_to_sequences([comment])
        padded = pad_sequences(seq, maxlen=MAX_LEN,padding='post')

        # Predict with DNN
        pred = model.predict(padded)[0].round(4)
        pred_bin = (pred > 0.4).astype(int)

        flagged_labels = [label for label, p in zip(TOXIC_LABELS, pred_bin) if p == 1]

        if flagged_labels:
            error_message = f"🚫 Your comment was flagged as inappropriate due to: {', '.join(flagged_labels)}."
        else:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO posts (username, comment, flagged) VALUES (?, ?, ?)",
                               (username, comment, None))
                conn.commit()
            success_message = "✅ Your comment has been posted."
            return redirect(url_for('home'))

    # Show all posts
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username, comment, flagged, timestamp FROM posts ORDER BY timestamp DESC")
        posts = cursor.fetchall()

    return render_template('index.html', posts=posts, error_message=error_message, success_message=success_message)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)