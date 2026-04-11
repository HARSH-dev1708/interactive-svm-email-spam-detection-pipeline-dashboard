import streamlit as st
import pickle
import string
from nltk.corpus import stopwords

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# Text cleaning function (same as training)
def transform_text(text):
    text = text.lower()
    text = text.split()

    y = []
    for word in text:
        if word.isalnum():
            y.append(word)

    final = []
    for word in y:
        if word not in stopwords.words('english') and word not in string.punctuation:
            final.append(word)

    return " ".join(final)

# UI
st.title("Email Spam Detection using SVM")

input_sms = st.text_area("Enter message")

if st.button("Predict"):
    # preprocess
    transformed_sms = transform_text(input_sms)

    # vectorize
    vector_input = tfidf.transform([transformed_sms]).toarray()

    # predict
    result = model.predict(vector_input)[0]

    # output
    if result == 1:
        st.header("Spam 🚫")
    else:
        st.header("Not Spam ✅")