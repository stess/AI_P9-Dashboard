import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import numpy as np

# Charger les modèles et les données
df = pd.read_csv("df_final.csv")

with open("Tf-idf_MLP-Classifier_model.pkl", "rb") as f:
    mlp_model = pickle.load(f)

with open("tfidf_model.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# Initialiser DistilBERT
model_path = "./distilbert_finetuned"  # Répertoire contenant le modèle fine-tuné
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Initialiser le pipeline avec le modèle fine-tuné
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Titre et description
st.title("Dashboard : Preuve de Concept Sentiment Analysis")
st.markdown("Ce dashboard présente les résultats de la classification des sentiments des tweets basés sur les modèles MLP et DistilBERT.")

# Analyse exploratoire
def exploratory_analysis(df):
    st.header("Analyse exploratoire")

    # Distribution de la longueur des tweets
    tweet_lengths = df['text'].apply(len)
    st.subheader("Distribution des longueurs de tweets")
    plt.figure(figsize=(10, 5))
    plt.hist(tweet_lengths, bins=30, color="#005a8c", edgecolor="black")
    plt.title("Distribution des longueurs de tweets")
    plt.xlabel("Longueur des tweets")
    plt.ylabel("Nombre de tweets")
    st.pyplot(plt)
    st.markdown("**Résumé du graphique :** La distribution montre que la majorité des tweets ont une longueur comprise entre 50 et 150 caractères.")


    st.subheader("Répartition des sentiments")
    sentiment_counts = df['target'].value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(
        sentiment_counts,
        labels=['Négatif', 'Positif'],
        autopct='%1.1f%%',
        colors=['#8c0005', '#005a8c'],
        startangle=90
    )
    plt.title("Répartition des sentiments")
    st.pyplot(plt)
    st.markdown("**Interprétation :** La répartition montre le pourcentage de tweets positifs et négatifs dans le dataset.")

    # WordCloud
    st.subheader("Nuage de mots")
    text = " ".join(df['text'])
    wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot(plt)
    st.markdown("**Interprétation :** Les mots les plus fréquents incluent des termes comme 'today', 'love', 'day' et 'now'.")

    st.subheader("Mots les plus fréquents")
    all_words = " ".join(df['text']).split()
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(25)
    words, counts = zip(*common_words)
    
    plt.figure(figsize=(8, 6))
    plt.bar(words, counts, color="#005a8c")
    plt.title("Top 25 des mots les plus fréquents")
    plt.ylabel("Fréquence")
    plt.xticks(rotation=45)
    st.pyplot(plt)
    st.markdown("**Interprétation :** Les mots les plus fréquents incluent des termes courants comme 'to', 'I', 'the' et 'a', qui sont souvent des stop words.")

    st.subheader("Comparaison des performances des modèles")

    models = ['MLP', 'DistilBERT']
    accuracy_scores = [0.6993, 0.8227]
    f1_scores = [0.6785, 0.8228]
    auc_scores = [0.7064, 0.8230]

    # Création du graphique groupé
    x = np.arange(len(models))
    width = 0.2  # Largeur des barres

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width, accuracy_scores, width, label='Accuracy', color='#005a8c')
    bars2 = ax.bar(x, f1_scores, width, label='F1 Score', color='#8c0005')
    bars3 = ax.bar(x + width, auc_scores, width, label='AUC', color='#ffb347')

    # Ajouter des labels
    ax.set_ylabel("Scores")
    ax.set_title("Performances des modèles")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    plt.ylim(0.6, 0.9)

    st.pyplot(fig)
    st.markdown("**Interprétation :** Ce graphique compare les performances des modèles MLP + Tf-idf et DistilBERT sur trois métriques : Accuracy, F1 Score et AUC. On observe que DistilBERT surpasse largement MLP dans toutes les métriques, indiquant une meilleure capacité à capturer les sentiments des tweets.")

exploratory_analysis(df)

# Sélection et prédiction
def prediction_section():
    st.header("Prédiction de sentiment")

    # Entrée utilisateur
    user_input = st.text_input("Entrez un tweet pour analyser son sentiment :", "I am so happy !", help="Cette zone est accessible aux lecteurs d'écran.")


    if st.button("Analyser"):
        # Prédiction avec MLP + Tf-idf
        user_tfidf = tfidf_vectorizer.transform([user_input])
        mlp_prediction = mlp_model.predict(user_tfidf)
        mlp_sentiment = "Positif" if mlp_prediction[0] == 1 else "Négatif"

        # Prédiction avec DistilBERT
        bert_result = sentiment_analyzer(user_input)[0]
        bert_sentiment = "Positif" if bert_result['label'] == "LABEL_1" else "Négatif"

        # Affichage des résultats
        st.write(f"**MLP :** Sentiment {mlp_sentiment}")
        st.write(f"**DistilBERT :** Sentiment {bert_sentiment} avec une confiance de {bert_result['score']:.2f}")

prediction_section()

# Accessibilité des graphiques
st.sidebar.header("Accessibilité")
st.sidebar.markdown("Ce dashboard prend en compte les critères d'accessibilité essentiels pour assurer une navigation inclusive.")
