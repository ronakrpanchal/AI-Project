import streamlit as st
import joblib
from PIL import Image
import os
import detoxify

# Define toxicity categories (used during model training)
TOXIC_LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MAX_LEN=200

# Get the absolute path to the models directory
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
models_dir = os.path.join(project_root, "models")

# Load the multi-label model and vectorizer
model = joblib.load(os.path.join(models_dir, "log_reg.pkl"))        # MultiOutputClassifier(LogisticRegression)
vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))

@st.cache_resource
def load_model():
    return detoxify.Detoxify('original')

mod = load_model()

# Initialize session state to store clean comments
if "comments" not in st.session_state:
    st.session_state.comments = []

# App UI
st.title("📱 Social Media Post")

st.markdown("""
<div style="background-color:#f0f2f6;padding:20px;border-radius:10px;">
""", unsafe_allow_html=True)

image = Image.open(os.path.join(current_dir, "my_image_2.png"))
st.image(image, caption="Today's Wordle of the day is truly a Sigma boi", use_column_width=True)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("### 💬 Comments Section")

# Input box for comments
user_comment = st.text_area("Add a comment...")

# Submit logic
if st.button("Post Comment"):
    if user_comment.strip() == "":
        st.warning("🚫 Comment cannot be empty.")
    else:
        vec = vectorizer.transform([user_comment])
        pred = model.predict(vec)[0]  # returns array of 0s/1s for all labels

        # # Check if any toxicity label is 1
        # if any(pred):
        #     flagged_labels = [label for label, p in zip(TOXIC_LABELS, pred) if p == 1]
        #     st.error(f"🚫 Your comment was flagged as inappropriate due to: {', '.join(flagged_labels)}.")
        # else:
        #     st.session_state.comments.append(user_comment)
        #     st.success("✅ Your comment has been posted.")
            
        pred = mod.predict(user_comment)
        flagged_labels = [key for key,value in pred.items() if value > 0.5]
        if flagged_labels:
            st.error(f"🚫 Your comment was flagged as inappropriate due to: {', '.join(flagged_labels)}.")
        else:
            st.session_state.comments.append(user_comment)
            st.success("✅ Your comment has been posted.")
            

# Display clean comments
if st.session_state.comments:
    st.markdown("### 📌 Recent Comments:")
    for i, c in enumerate(reversed(st.session_state.comments)):
        st.markdown(f"- {c}")