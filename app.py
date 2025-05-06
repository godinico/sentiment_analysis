import os
import pickle
import re
import uuid
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from flask import Flask, request, jsonify
from opencensus.ext.azure.log_exporter import AzureLogHandler
import logging
from functools import wraps

# Configuration du logger
INSTRUMENTATION_KEY = "InstrumentationKey=275b0505-425b-45cb-8ea5-5d5a64304e97"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(AzureLogHandler(connection_string=INSTRUMENTATION_KEY))

app = Flask(__name__)

# Chargement des ressources
model = tf.keras.models.load_model('model_LSTM.h5')
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Utilitaires
def preprocess_text(text):
    """Prétraite le texte pour l'analyse de sentiment"""
    text = text.lower() # Convertir en minuscules
    text = re.sub(r"@\w+", "", text)  # Supprimer les mentions
    text = re.sub(r"http\S+|www\S+|https\S+", "URL", text, flags=re.MULTILINE)  # Remplacer URLs
    text = re.sub(r"#(\w+)", r"\1", text)  # Gérer les hashtags
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # Réduire les répétitions
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Supprimer caractères spéciaux
    
    # Convertir en séquence pour le modèle
    sequence = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequence, maxlen=100)

def predict_sentiment(text):
    """Prédit le sentiment d'un texte"""
    input_data = preprocess_text(text)
    predictions = model.predict(input_data)
    return float(predictions[0][0])

def requires_json(f):
    """Décorateur pour vérifier si la requête contient du JSON"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not request.is_json:
            return jsonify({"error": "La requête doit être en format JSON"}), 400
        return f(*args, **kwargs)
    return decorated_function

def log_event(event_type, properties):
    """Fonction centralisée pour les logs"""
    if event_type == "info":
        logger.info(f"{properties.get('message', '')}", extra={"custom_dimensions": properties})
    elif event_type == "warning":
        logger.warning(f"{properties.get('message', '')}", extra={"custom_dimensions": properties})
    elif event_type == "error":
        logger.error(f"{properties.get('message', '')}", extra={"custom_dimensions": properties})

# Routes
@app.route('/health', methods=['GET'])
def health_check():
    """Point de terminaison pour vérifier la santé de l'API"""
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
@requires_json
def predict():
    """Point de terminaison pour prédire le sentiment d'un texte"""
    data = request.get_json()
    
    if 'text' not in data:
        return jsonify({"error": "Le champ 'text' est requis"}), 400
    
    text = data['text']
    
    try:
        request_id = str(uuid.uuid4())
        sentiment = predict_sentiment(text)
        sentiment_label = "positif" if sentiment > 0.5 else "négatif"
        
        # Logs
        log_event("info", {
            "message": "Prediction",
            "request_id": request_id,
            "text": text,
            "sentiment": sentiment,
            "label": sentiment_label
        })
        
        return jsonify({
            "request_id": request_id,
            "text": text,
            "sentiment": sentiment,
            "sentiment_label": sentiment_label
        })
    
    except Exception as e:
        log_event("error", {"message": f"Error during prediction: {str(e)}"})
        return jsonify({"error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
@requires_json
def feedback():
    """Point de terminaison unifié pour les retours utilisateurs"""
    data = request.get_json()
    
    # Validation des champs
    required_fields = ['text', 'prediction']
    optional_fields = ['request_id', 'is_correct', 'user_id']
    
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Le champ '{field}' est requis"}), 400
    
    # Préparation des données
    properties = {
        "request_id": data.get('request_id', str(uuid.uuid4())),
        "text": data['text'],
        "prediction": data['prediction'],
        "is_correct": data.get('is_correct', False),
        "timestamp": time.time(),
        "user_agent": request.headers.get('User-Agent', 'Unknown'),
        "message": "Feedback received"
    }
    
    # Ajouter des champs optionnels s'ils existent
    for field in optional_fields:
        if field in data and field not in properties:
            properties[field] = data[field]
    
    # Log selon le type de feedback
    if data.get('is_correct', False):
        log_event("info", {**properties, "message": "Correct prediction confirmed"})
        status_message = "Merci pour votre confirmation !"
    else:
        log_event("warning", {**properties, "message": "Incorrect prediction reported"})
        status_message = "Merci pour votre retour !"
    
    return jsonify({
        "status": "feedback_received",
        "is_correct": data.get('is_correct', False),
        "message": status_message
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)