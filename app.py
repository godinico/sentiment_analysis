import os
import pickle
import mlflow
import re
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

# Configuration du logger pour Application Insights
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Si vous avez une clé d'instrumentation Application Insights
# Sinon, commentez cette ligne ou définissez la clé via une variable d'environnement
# INSTRUMENTATION_KEY = os.environ.get("APPINSIGHTS_INSTRUMENTATIONKEY", "")
# if INSTRUMENTATION_KEY:
#     logger.addHandler(AzureLogHandler(connection_string=f'InstrumentationKey={INSTRUMENTATION_KEY}'))

app = Flask(__name__)

# Charger le modèle
model = tf.keras.models.load_model('model_LSTM.h5')

# Charger le tokenizer 
with open('trained_tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Fonction de prétraitement de texte
def preprocess_text(text):
    # 1. Convertir en minuscules
    text = text.lower()

    # 2. Supprimer les mentions (@user)
    text = re.sub(r"@\w+", "", text)

    # 3. Remplacer les URLs par "URL"
    text = re.sub(r"http\S+|www\S+|https\S+", "URL", text, flags=re.MULTILINE)

    # 4. Gérer les hashtags (#word → word)
    text = re.sub(r"#(\w+)", r"\1", text)

    # 5. Réduire les répétitions de lettres (ex: coooool → cool)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # 6. Supprimer la ponctuation excessive et les caractères spéciaux
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Convertir le texte en séquence
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    
    return padded_sequence

# Fonction de prédiction
def predict_sentiment(text):
    # Prétraiter le texte
    input_data = preprocess_text(text)
    
    # Effectuer la prédiction avec le modèle
    predictions = model.predict(input_data)
    
    # Convertir la sortie en sentiment (0 à 1)
    sentiment_score = float(predictions[0][0])
    
    return sentiment_score

@app.route('/health', methods=['GET'])
def health_check():
    """Point de terminaison pour vérifier la santé de l'API"""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    """Point de terminaison pour prédire le sentiment d'un tweet"""
    # Vérifier si la requête contient du JSON
    if not request.is_json:
        return jsonify({"error": "La requête doit être en format JSON"}), 400
    
    # Obtenir le texte du tweet
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "Le champ 'text' est requis"}), 400
    
    text = data['text']
    
    try:
        # Prédire le sentiment
        sentiment = predict_sentiment(text)
        
        # Logger la prédiction
        logger.info(f"Prediction: text='{text}', sentiment={sentiment}")
        
        # Retourner la prédiction
        return jsonify({
            "text": text,
            "sentiment": sentiment,
            "sentiment_label": "positif" if sentiment > 0.5 else "négatif"
        })
    
    except Exception as e:
        # Logger l'erreur
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/report', methods=['POST'])
def report():
    """Point de terminaison pour signaler une prédiction incorrecte"""
    # Vérifier si la requête contient du JSON
    if not request.is_json:
        return jsonify({"error": "La requête doit être en format JSON"}), 400
    
    # Obtenir les données
    data = request.get_json()
    if 'text' not in data or 'prediction' not in data:
        return jsonify({"error": "Les champs 'text' et 'prediction' sont requis"}), 400
    
    text = data['text']
    prediction = data['prediction']
    
    # Logger la mauvaise prédiction
    logger.warning(f"Incorrect prediction reported: text='{text}', prediction={prediction}")
    
    return jsonify({"status": "reported", "message": "Merci pour votre retour !"})

if __name__ == '__main__':
    # Pour le développement local
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)