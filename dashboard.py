import streamlit as st
import pandas as pd
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re

# -------------------- TEXT CLEANING FUNCTION --------------------
def transform_text(text):
    text = text.lower()
    words = text.split()
    
    # Stricter cleaning from your notebook
    final = []
    stop_words = set(stopwords.words('english'))
    for word in words:
        if word.isalnum() and word not in stop_words and word not in string.punctuation:
            final.append(word)
    return " ".join(final)


# -------------------- PAGE CONFIG --------------------
st.set_page_config(layout="wide")

# -------------------- SESSION STATE --------------------
if "df" not in st.session_state:
    st.session_state.df = None

if "clean_df" not in st.session_state:
    st.session_state.clean_df = None


# -------------------- SIDEBAR --------------------
st.sidebar.title("Data Source")

uploaded_file = st.sidebar.file_uploader("Upload CSV")

if st.sidebar.button("Use Default Dataset"):
    st.session_state.df = pd.read_csv("spam.csv", encoding='latin-1')
    # st.session_state.df = pd.read_csv("spam.csv")

if uploaded_file is not None:
    st.session_state.df = pd.read_csv(uploaded_file)

if st.session_state.df is not None:
    st.sidebar.success("Dataset Loaded ✅")


# -------------------- TITLE --------------------
st.title("📊 Interactive ML Pipeline Dashboard")

# -------------------- TABS --------------------
tabs = st.tabs([
    "Data & EDA",
    "Cleaning & Engineering",
    "Feature Selection",
    "Model Training",
    "Performance"
])


# ==================== TAB 1 ====================
with tabs[0]:
    st.header("📁 Data Overview & EDA")

    df = st.session_state.df

    if df is not None:
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        st.write("### Shape of Data")
        st.write(df.shape)

        st.write("### Columns in Dataset")
        st.write(df.columns)

        # Spam vs Ham
        if 'label' in df.columns:
            st.write("### Spam vs Ham Distribution")
            st.bar_chart(df['label'].value_counts())

        elif 'v1' in df.columns:
            st.write("### Spam vs Ham Distribution")
            st.bar_chart(df['v1'].value_counts())

    else:
        st.warning("Please upload or select dataset")


# ==================== TAB 2 ====================
with tabs[1]:
    st.header("🧹 Cleaning & Engineering")
    df = st.session_state.df

    if df is not None:
        # Identify the correct column
        text_col = 'v2' if 'v2' in df.columns else ('message' if 'message' in df.columns else None)
        
        if text_col:
            if st.button("Apply Cleaning"):
                with st.spinner("Cleaning text... please wait."):
                    # Apply transformation
                    df['clean_text'] = df[text_col].apply(transform_text)
                    st.session_state.clean_df = df
                st.success("Text cleaned successfully!")
            
            # This ensures the table stays visible after the button is clicked
            if st.session_state.clean_df is not None:
                st.write("### Cleaned Data Preview")
                st.dataframe(st.session_state.clean_df[[text_col, 'clean_text']].head(10))
        else:
            st.error("Could not find a text column (v2 or message).")
    else:
        st.warning("Load dataset first")

# ==================== TAB 3 ====================
with tabs[2]:
    st.header("🎯 Feature Selection")
    st.write("Coming next step...")


# ==================== TAB 4 ====================
with tabs[3]:
    st.header("🤖 Model Training")
    st.write("Coming next step...")


# ==================== TAB 5 ====================
with tabs[4]:
    st.header("📈 Performance")
    st.write("Coming next step...")