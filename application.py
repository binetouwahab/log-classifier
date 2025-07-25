import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np

nltk.download('stopwords')
nltk.download('wordnet')

# ✅ PAGE CONFIG
st.set_page_config(page_title="Classification des Logs", page_icon="🔍", layout="centered")

# ✅ STYLE CSS AVEC ANIMATIONS
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #fafafa;
        }
        .title {
            text-align: center;
            font-size: 2.7em;
            font-weight: bold;
            color: #61dafb;
            animation: fadeIn 2s ease-in-out;
        }
        .subheader {
            text-align: center;
            font-size: 1.1em;
            color: #cccccc;
            animation: slideIn 2s ease-in-out;
        }
        .stTextArea textarea {
            background-color: #1c1e24;
            color: white;
            border-radius: 10px;
            font-size: 1.1em;
            transition: 0.3s;
        }
        .stTextArea textarea:focus {
            border: 2px solid #61dafb;
        }
        .stButton>button {
            background-color: #61dafb;
            color: black;
            border-radius: 10px;
            font-size: 1.1em;
            padding: 0.6em 1.5em;
            transition: 0.4s;
        }
        .stButton>button:hover {
            background-color: #21a1f1;
            color: white;
            transform: scale(1.05);
        }
        .pred-box {
            background-color: #1c3c29;
            color: #00ff88;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
            font-size: 1.3em;
            font-weight: bold;
            animation: fadeIn 1.5s ease-in-out;
        }
        @keyframes fadeIn {
            0% {opacity: 0;}
            100% {opacity: 1;}
        }
        @keyframes slideIn {
            0% {transform: translateY(-20px); opacity:0;}
            100% {transform: translateY(0); opacity:1;}
        }
    </style>
""", unsafe_allow_html=True)

# ✅ PRETRAITEMENT
def nettoyer_texte(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'http\S+', '', text)
    return text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def pretraitement(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# ✅ CHARGEMENT ET ENTRAINEMENT
@st.cache_data
def entrainer_modele():
    with open("output_0.1.log", "r") as f:
        lines = f.readlines()

    categories, logs = [], []
    for line in lines[1:]:
        parts = line.split(",", 1)
        if len(parts) == 2:
            categories.append(parts[0].strip())
            logs.append(parts[1].strip())
        else:
            categories.append("unknown")
            logs.append(line.strip())

    df = pd.DataFrame({"category": categories, "log": logs})
    df['clean_log'] = df['log'].apply(nettoyer_texte).apply(pretraitement)

    vectorizer = TfidfVectorizer(max_features=2000, min_df=2, max_df=0.8)
    X = vectorizer.fit_transform(df['clean_log'])
    y = df['category']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, acc, y

model, vectorizer, accuracy, y = entrainer_modele()

# ✅ INTERFACE
st.markdown("<div class='title'>🔍 Classification des Logs</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subheader'>📊 Précision du modèle sur Test : {accuracy*100:.2f}%</div>", unsafe_allow_html=True)

text_input = st.text_area("✏️ Entrez un journal à analyser", height=120)

if st.button("🔮 Prédire"):
    if text_input.strip():
        with st.spinner("⏳ Analyse du log en cours..."):
            time.sleep(1.5)  # Effet de chargement
            text_clean = pretraitement(nettoyer_texte(text_input))
            vec = vectorizer.transform([text_clean])
            pred = model.predict(vec)[0]
            
            # ✅ TOP 3 DES PROBABILITÉS
            probs = model.predict_proba(vec)[0]
            classes = model.classes_
            top3_idx = np.argsort(probs)[-3:][::-1]
            
        st.markdown(f"<div class='pred-box'>✅ Catégorie prédite : {pred}</div>", unsafe_allow_html=True)
        
        st.subheader("📈 Top 3 des catégories probables")
        for idx in top3_idx:
            st.progress(float(probs[idx]))
            st.write(f"**{classes[idx]}** : {probs[idx]*100:.2f}%")
    else:
        st.warning("⚠️ Veuillez entrer un texte de log.")
