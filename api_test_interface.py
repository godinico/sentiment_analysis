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

# Zone de saisie du tweet
tweet = st.text_area("", height=68)

# Bouton d'analyse
if st.button("Détecter le sentiment"):
    if tweet:
        # Appel à l'API
        try:
            response = requests.post(f"{API_URL}/predict", json={"text": tweet})
            if response.status_code == 200:
                result = response.json()
                sentiment = result["sentiment"]
                label = result["sentiment_label"]
                
                # Afficher le résultat
                st.subheader("Résultat de l'analyse :")
                if label == "positif":
                    st.success(f"Le sentiment associé à ce tweet est considéré comme POSITIF (score : {sentiment:.2f})")
                else:
                    st.error(f"Le sentiment associé à ce tweet est considéré comme NÉGATIF (score : {sentiment:.2f})")
                
                # Demander une validation
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Prédiction correcte"):
                        st.success("Merci pour votre retour!")
                
                with col2:
                    if st.button("❌ Prédiction incorrecte"):
                        # Envoyer le rapport
                        report_response = requests.post(f"{API_URL}/report", 
                                                     json={"text": tweet, "prediction": sentiment})
                        if report_response.status_code == 200:
                            st.info("Merci d'avoir signalé cette erreur. Nous utiliserons ce retour pour améliorer notre modèle.")
                        else:
                            st.warning("Impossible d'envoyer le rapport d'erreur.")
            else:
                st.error(f"Erreur lors de l'appel à l'API : {response.status_code}")
        except Exception as e:
            st.error(f"Erreur : {str(e)}")
    else:
        st.warning("Veuillez entrer un tweet à analyser.")