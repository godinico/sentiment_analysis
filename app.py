import os
import pickle
import re
import uuid
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from flask import Flask, request, jsonify
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging

# Configuration du logger pour Application Insights
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
INSTRUMENTATION_KEY = "InstrumentationKey=275b0505-425b-45cb-8ea5-5d5a64304e97"
logger.addHandler(AzureLogHandler(connection_string=INSTRUMENTATION_KEY))

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
        # Générer un ID unique pour cette prédiction
        request_id = str(uuid.uuid4())
        
        # Prédire le sentiment
        sentiment = predict_sentiment(text)
        
        # Déterminer le label du sentiment
        sentiment_label = "positif" if sentiment > 0.5 else "négatif"
        
        # Logger la prédiction
        logger.info(f"Prediction: request_id={request_id}, text='{text}', sentiment={sentiment}, label={sentiment_label}")
        
        # Retourner la prédiction avec l'ID de requête
        return jsonify({
            "request_id": request_id,
            "text": text,
            "sentiment": sentiment,
            "sentiment_label": sentiment_label
        })
    
    except Exception as e:
        # Logger l'erreur
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Point de terminaison pour recevoir le feedback sur une prédiction (correcte ou incorrecte)"""
    # Vérifier si la requête contient du JSON
    if not request.is_json:
        return jsonify({"error": "La requête doit être en format JSON"}), 400
    
    # Obtenir les données
    data = request.get_json()
    required_fields = ['request_id', 'text', 'prediction', 'is_correct']
    
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Le champ '{field}' est requis"}), 400
    
    request_id = data['request_id']
    text = data['text']
    prediction = data['prediction']
    is_correct = data['is_correct']
    
    # Métadonnées supplémentaires
    timestamp = time.time()
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    # Créer un dictionnaire de propriétés pour la trace
    properties = {
        "request_id": request_id,
        "text": text,
        "prediction": prediction,
        "is_correct": is_correct,
        "timestamp": timestamp,
        "user_agent": user_agent
    }
    
    # Ajouter des métadonnées supplémentaires si disponibles
    if 'user_id' in data:
        properties["user_id"] = data['user_id']
    
    if 'expected_prediction' in data and not is_correct:
        properties["expected_prediction"] = data['expected_prediction']
    
    # Logger le feedback selon qu'il soit positif ou négatif
    if is_correct:
        logger.info("Correct prediction confirmed", extra={"custom_dimensions": properties})
        status_message = "Merci pour votre confirmation !"
    else:
        logger.warning("Incorrect prediction reported", extra={"custom_dimensions": properties})
        status_message = "Merci pour votre retour !"
    
    return jsonify({
        "status": "feedback_received", 
        "is_correct": is_correct,
        "message": status_message
    })

# Garder la route /report pour la compatibilité avec l'ancien code
@app.route('/report', methods=['POST'])
def report():
    """Point de terminaison pour signaler une prédiction incorrecte (route maintenue pour compatibilité)"""
    # Vérifier si la requête contient du JSON
    if not request.is_json:
        return jsonify({"error": "La requête doit être en format JSON"}), 400
    
    # Obtenir les données
    data = request.get_json()
    if 'text' not in data or 'prediction' not in data:
        return jsonify({"error": "Les champs 'text' et 'prediction' sont requis"}), 400
    
    text = data['text']
    prediction = data['prediction']
    
    # Générer un ID pour cette requête
    request_id = str(uuid.uuid4())
    
    # Métadonnées supplémentaires
    timestamp = time.time()
    user_agent = request.headers.get('User-Agent', 'Unknown')
    
    # Créer un dictionnaire de propriétés pour la trace
    properties = {
        "request_id": request_id,
        "text": text,
        "prediction": prediction,
        "is_correct": False,
        "timestamp": timestamp,
        "user_agent": user_agent
    }
    
    # Logger la mauvaise prédiction
    logger.warning("Incorrect prediction reported", extra={"custom_dimensions": properties})
    
    return jsonify({"status": "reported", "message": "Merci pour votre retour !"})

if __name__ == '__main__':
    # Pour le développement local
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)