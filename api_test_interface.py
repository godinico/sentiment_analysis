import streamlit as st
import requests

API_URL = "https://tweet-sentiment-analysis-api-cwgzfag8caf0bda4.westeurope-01.azurewebsites.net"

st.markdown("""
    <div style="background-color: #800020; padding: 2px; border-radius: 15px; text-align: center;">
        <h1 style="color: white; margin: 0;">Détecteur de sentiments</h1>
    </div>
    """, unsafe_allow_html=True)

# Espace après le titre
st.write("")

st.subheader("Saisir un tweet à analyser (en anglais) :")

# Initialiser les variables d'état dans la session
if 'request_id' not in st.session_state:
    st.session_state.request_id = None
if 'tweet' not in st.session_state:
    st.session_state.tweet = ""
if 'sentiment' not in st.session_state:
    st.session_state.sentiment = None
if 'feedback_sent' not in st.session_state:
    st.session_state.feedback_sent = False

# Zone de saisie du tweet avec un label explicite mais masqué
tweet = st.text_area(
    "Texte du tweet", 
    height=68, 
    value=st.session_state.tweet,
    label_visibility="collapsed"  # Cache le label mais le garde pour l'accessibilité
)

# Fonction pour réinitialiser après un feedback
def reset_feedback():
    st.session_state.feedback_sent = True

# Bouton d'analyse
if st.button("Détecter le sentiment") or (tweet != st.session_state.tweet and tweet):
    if tweet:
        st.session_state.tweet = tweet
        st.session_state.feedback_sent = False
        
        # Appel à l'API
        try:
            response = requests.post(f"{API_URL}/predict", json={"text": tweet})
            if response.status_code == 200:
                result = response.json()
                st.session_state.request_id = result["request_id"]
                st.session_state.sentiment = result["sentiment"]
                label = result["sentiment_label"]
                
                # Afficher le résultat
                st.subheader("Résultat de l'analyse :")
                if label == "positif":
                    st.success(f"Le sentiment associé à ce tweet est considéré comme POSITIF (score : {st.session_state.sentiment:.2f})")
                else:
                    st.error(f"Le sentiment associé à ce tweet est considéré comme NÉGATIF (score : {st.session_state.sentiment:.2f})")
            else:
                st.error(f"Erreur lors de l'appel à l'API : {response.status_code}")
        except Exception as e:
            st.error(f"Erreur : {str(e)}")
    else:
        st.warning("Veuillez entrer un tweet à analyser.")

# Affichage des boutons de feedback uniquement si une prédiction a été faite et pas encore de feedback
if st.session_state.request_id and not st.session_state.feedback_sent:
    st.write("Cette prédiction vous semble-t-elle correcte ?")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ Prédiction correcte"):
            feedback_response = requests.post(f"{API_URL}/feedback", 
                                          json={
                                              "request_id": st.session_state.request_id,
                                              "text": st.session_state.tweet,
                                              "prediction": st.session_state.sentiment,
                                              "is_correct": True
                                          })
            if feedback_response.status_code == 200:
                st.success("Merci pour votre confirmation!")
                reset_feedback()
            else:
                st.warning(f"Impossible d'envoyer le feedback: {feedback_response.status_code}")
    
    with col2:
        if st.button("❌ Prédiction incorrecte"):
            feedback_response = requests.post(f"{API_URL}/feedback", 
                                          json={
                                              "request_id": st.session_state.request_id,
                                              "text": st.session_state.tweet,
                                              "prediction": st.session_state.sentiment,
                                              "is_correct": False
                                          })
            if feedback_response.status_code == 200:
                st.info("Merci d'avoir signalé cette erreur. Nous utiliserons ce retour pour améliorer notre modèle.")
                reset_feedback()
            else:
                st.warning(f"Impossible d'envoyer le feedback: {feedback_response.status_code}")

# Option pour effacer et recommencer
if st.session_state.feedback_sent:
    if st.button("Analyser un nouveau tweet"):
        st.session_state.request_id = None
        st.session_state.tweet = ""
        st.session_state.sentiment = None
        st.session_state.feedback_sent = False
        st.rerun()