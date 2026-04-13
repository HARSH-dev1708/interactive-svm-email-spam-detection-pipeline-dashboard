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
# ==================== TAB 3 ====================
with tabs[2]:
    st.header("🎯 Feature Selection (TF-IDF Vectorization)")
    
    # Check if the cleaning step was done
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        
        # 1. Label Mapping (Necessary for SVM)
        # Standardizing labels to 0 and 1
        target_col = 'v1' if 'v1' in df.columns else 'label'
        if 'label_num' not in df.columns:
            df['label_num'] = df[target_col].map({'ham': 0, 'spam': 1})
        
        st.subheader("1. Label Encoding")
        st.write("Converting text labels to numeric: `ham` → 0, `spam` → 1")
        st.dataframe(df[[target_col, 'label_num']].head())

        st.divider()

        # 2. TF-IDF Configuration
        st.subheader("2. TF-IDF Vectorization")
        st.info("TF-IDF (Term Frequency-Inverse Document Frequency) converts words into statistical scores based on their importance.")
        
        # User can experiment with vocabulary size
        max_features = st.slider("Select Max Vocabulary Size (Features)", 500, 5000, 3000)
        
        if st.button("Generate Feature Matrix"):
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            with st.spinner("Calculating TF-IDF scores..."):
                tfidf = TfidfVectorizer(max_features=max_features)
                X = tfidf.fit_transform(df['clean_text']).toarray()
                y = df['label_num'].values
                
                # Store in session state for the Training Tab (Tab 4)
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.tfidf = tfidf
                
                st.success(f"Feature Matrix Created! Shape: {X.shape}")
                
                # Show the matrix as a DataFrame with actual word names
                st.write("### Top Features Preview")
                feature_names = tfidf.get_feature_names_out()
                sample_matrix = pd.DataFrame(X, columns=feature_names).head(10)
                st.dataframe(sample_matrix)
    else:
        st.warning("⚠️ Please complete 'Tab 2: Cleaning & Engineering' first.")


# ==================== TAB 4 ====================
with tabs[3]:
    st.header("🤖 Model Training (SVM)")
    
    # Check if Feature Matrix exists from Tab 3
    if "X" in st.session_state and "y" in st.session_state:
        X = st.session_state.X
        y = st.session_state.y
        
        st.write("### 1. Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 50, 20) / 100
            random_state = st.number_input("Random State", value=2)
            
        with col2:
            kernel = st.selectbox("SVM Kernel", ["linear", "rbf", "poly", "sigmoid"])
            C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)

        if st.button("🚀 Train SVM Model"):
            from sklearn.model_selection import train_test_split
            from sklearn.svm import SVC
            import time

            # Step 1: Split Data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Step 2: Initialize & Train
            model = SVC(kernel=kernel, C=C)
            
            start_time = time.time()
            with st.spinner(f"Training SVM with {kernel} kernel..."):
                model.fit(X_train, y_train)
            end_time = time.time()
            
            # Step 3: Store results in session state
            st.session_state.model = model
            st.session_state.test_data = (X_test, y_test)
            st.session_state.train_time = end_time - start_time
            
            st.success(f"Model Trained in {st.session_state.train_time:.2f} seconds! ✅")
            
            # Show basic result
            y_pred = model.predict(X_test)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_test, y_pred)
            st.metric("Model Accuracy", f"{acc*100:.2f}%")
            
    else:
        st.warning("⚠️ Please generate the Feature Matrix in Tab 3 first!")


# ==================== TAB 5 ====================
with tabs[4]:
    st.header("📈 Performance Analysis")

    if "model" in st.session_state:
        model = st.session_state.model
        X_test, y_test = st.session_state.test_data
        tfidf = st.session_state.tfidf

        # --- SECTION 1: METRICS ---
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import confusion_matrix, classification_report
        import matplotlib.pyplot as plt
        import seaborn as sns

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Ham', 'Spam'], 
                        yticklabels=['Ham', 'Spam'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)

        with col2:
            st.write("### Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

        st.divider()

        # --- SECTION 2: LIVE TESTER ---
        st.write("### 🧪 Live Spam Detector")
        st.info("Type a message below to test the trained SVM model!")
        
        user_input = st.text_area("Enter Email/SMS content here:")

        if st.button("Predict"):
            if user_input:
                # 1. Clean (using our function from earlier)
                cleaned_input = transform_text(user_input)
                
                # 2. Vectorize (using the tfidf object we saved)
                vector_input = tfidf.transform([cleaned_input]).toarray()
                
                # 3. Predict
                prediction = model.predict(vector_input)[0]
                
                # 4. Result
                if prediction == 1:
                    st.error("🚨 This is SPAM!")
                else:
                    st.success("✅ This is HAM (Safe).")
            else:
                st.warning("Please enter some text first.")
                
    else:
        st.warning("⚠️ Train the model in Tab 4 first to see performance metrics.")