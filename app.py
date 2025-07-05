import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import re
import spacy
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def tokenize_and_lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.text not in stop_words])

def preprocess(text):
    return tokenize_and_lemmatize(clean_text(text))

model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

reply_templates = {
    "Praise": "Thank you for your kind words! 😊",
    "Support": "We appreciate your support!",
    "Constructive Criticism": "Thanks! We'll work on improving that.",
    "Hate/Abuse": "We’re monitoring all comments and maintaining respect.",
    "Threat": "This has been flagged for moderation.",
    "Emotional": "We’re touched this resonated with you.",
    "Irrelevant/Spam": "This seems off-topic. We'll review it.",
    "Question/Suggestion": "Thanks! We'll consider it in future updates."
}

st.title("🧠 Comment Categorization & Reply Assistant")

user_input = st.text_area("Enter a comment for classification:")

if st.button("Classify"):
    if user_input.strip():
        processed = preprocess(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        st.success(f"Predicted Category: {prediction}")
        st.info(f"Suggested Reply: {reply_templates.get(prediction)}")
    else:
        st.warning("Please enter a comment.")

st.header("📂 Upload CSV for Batch Classification")
uploaded_file = st.file_uploader("Upload a CSV with 'comment' column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if "comment" in df.columns:
        df["processed"] = df["comment"].apply(preprocess)
        df["predicted_category"] = model.predict(vectorizer.transform(df["processed"]))
        st.write("### Classified Comments")
        st.dataframe(df[["comment", "predicted_category"]])
        st.download_button("📥 Download Results", df.to_csv(index=False).encode(), file_name="classified_comments.csv")

        st.write("### 📊 Category Distribution")
        dist = df["predicted_category"].value_counts()
        fig, ax = plt.subplots()
        dist.plot(kind="bar", color="skyblue", ax=ax)
        ax.set_ylabel("Count")
        ax.set_title("Predicted Comment Category Distribution")
        st.pyplot(fig)
    else:
        st.error("CSV must contain a 'comment' column.")
