
import streamlit as st
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Label mapping
label_map = {0: "😡 Negative", 1: "😐 Neutral", 2: "😊 Positive"}

st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")
st.title("💬 Customer Review Sentiment Analyzer")
st.write("🔍 **Analyze your product review to detect sentiment**")

review_text = st.text_area("📝 Enter your review below:", height=150)

if st.button("🔮 Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        review_vec = vectorizer.transform([review_text])
        prediction = model.predict(review_vec)[0]
        sentiment = label_map[prediction]
        st.success(f"✅ **Predicted Sentiment:** {sentiment}")
