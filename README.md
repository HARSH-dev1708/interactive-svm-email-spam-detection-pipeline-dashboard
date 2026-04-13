# SpamShield: Interactive SVM Pipeline Visualizer

**SpamShield** is an end-to-end Machine Learning dashboard built with Streamlit. It allows users to visualize the entire lifecycle of an Email Spam Detection system—from raw data exploration to real-time predictions using Support Vector Machines (SVM).

## Project Overview
Most ML projects are "black boxes." This project aims to break down that box by providing an interactive interface for every step of the pipeline. You can tune hyperparameters like **C** and **Kernels** live and see how they affect the model's performance.

---

## Key Features

* **Data & EDA Tab:** Explore the dataset, check the Spam/Ham ratio, and see raw email samples.
* **Cleaning & Engineering Tab:** Watch how raw text is transformed into clean tokens using NLTK (lowercasing, stopword removal, and punctuation stripping).
* **Feature Selection Tab:** Visualize the **TF-IDF Matrix** and see how words are converted into mathematical scores.
* **Model Training Tab:** Interactively train an **SVM** model. Adjust the `test_size`, `random_state`, and SVM-specific hyperparameters like `C` and `Kernel`.
* **Performance Analysis Tab:** Evaluate the model using a **Confusion Matrix** and **Classification Report**, then test it with your own custom messages!

---

## Tech Stack
* **Language:** Python
* **Framework:** Streamlit
* **ML Library:** Scikit-Learn
* **NLP:** NLTK
* **Visualization:** Matplotlib, Seaborn, Pandas

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HARSH-dev1708/spamshield-svm-visualizer

2. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

3. **Run the Dashboard:**
    ```bash
    streamlit run dashboard.py
