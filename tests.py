import unittest
import json
from unittest.mock import patch, MagicMock
from app import app, predict_sentiment

class SentimentAPITestCase(unittest.TestCase):
    """Tests unitaires pour l'API d'analyse de sentiment."""

    def setUp(self):
        """Configuration initiale avant chaque test."""
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_check(self):
        """Tester la route /health."""
        response = self.app.get('/health')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
    
    @patch('app.predict_sentiment')
    def test_predict_valid_input(self, mock_predict):
        """Tester la route /predict avec une entrée valide."""
        # Configurer le mock pour predict_sentiment
        mock_predict.return_value = 0.8
        
        # Envoyer une requête de test
        response = self.app.post(
            '/predict',
            data=json.dumps({'text': 'Je suis très content!'}),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        # Vérifier les résultats
        self.assertEqual(response.status_code, 200)
        self.assertIn('request_id', data)
        self.assertEqual(data['text'], 'Je suis très content!')
        self.assertEqual(data['sentiment'], 0.8)
        self.assertEqual(data['sentiment_label'], 'positif')
        
        # Vérifier que predict_sentiment a été appelé avec le bon argument
        mock_predict.assert_called_once_with('Je suis très content!')
    
    def test_predict_invalid_request_format(self):
        """Tester la route /predict avec un format de requête invalide."""
        response = self.app.post(
            '/predict',
            data='Ceci n\'est pas du JSON',
            content_type='text/plain'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'La requête doit être en format JSON')
    
    def test_predict_missing_text_field(self):
        """Tester la route /predict avec un champ 'text' manquant."""
        response = self.app.post(
            '/predict',
            data=json.dumps({'autre_champ': 'valeur'}),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
        self.assertEqual(data['error'], "Le champ 'text' est requis")
    
    @patch('app.predict_sentiment')
    def test_predict_negative_sentiment(self, mock_predict):
        """Tester la route /predict avec un sentiment négatif."""
        # Configurer le mock pour predict_sentiment
        mock_predict.return_value = 0.2
        
        # Envoyer une requête de test
        response = self.app.post(
            '/predict',
            data=json.dumps({'text': 'Je suis très déçu!'}),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        # Vérifier les résultats
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['sentiment'], 0.2)
        self.assertEqual(data['sentiment_label'], 'négatif')
    
    @patch('app.predict_sentiment')
    def test_predict_exception_handling(self, mock_predict):
        """Tester la gestion des exceptions dans la route /predict."""
        # Configurer le mock pour lever une exception
        mock_predict.side_effect = Exception("Erreur de test")
        
        # Envoyer une requête de test
        response = self.app.post(
            '/predict',
            data=json.dumps({'text': 'test'}),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        # Vérifier les résultats
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', data)
        self.assertEqual(data['error'], "Erreur de test")
    
    def test_feedback_valid_input(self):
        """Tester la route /feedback avec une entrée valide."""
        response = self.app.post(
            '/feedback',
            data=json.dumps({
                'request_id': '123e4567-e89b-12d3-a456-426614174000',
                'text': 'Ce produit est génial',
                'prediction': 0.9,
                'is_correct': True
            }),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'feedback_received')
        self.assertEqual(data['is_correct'], True)
        self.assertEqual(data['message'], 'Merci pour votre confirmation !')
    
    def test_feedback_incorrect_prediction(self):
        """Tester la route /feedback avec une prédiction incorrecte."""
        response = self.app.post(
            '/feedback',
            data=json.dumps({
                'request_id': '123e4567-e89b-12d3-a456-426614174000',
                'text': 'Ce produit est terrible',
                'prediction': 0.8,
                'is_correct': False,
                'expected_prediction': 0.2
            }),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'feedback_received')
        self.assertEqual(data['is_correct'], False)
        self.assertEqual(data['message'], 'Merci pour votre retour !')
    
    def test_feedback_missing_fields(self):
        """Tester la route /feedback avec des champs manquants."""
        response = self.app.post(
            '/feedback',
            data=json.dumps({
                'request_id': '123e4567-e89b-12d3-a456-426614174000',
                'text': 'Test'
                # Champs 'prediction' et 'is_correct' manquants
            }),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
    
    def test_report_valid_input(self):
        """Tester la route /report avec une entrée valide."""
        response = self.app.post(
            '/report',
            data=json.dumps({
                'text': 'Ce test ne fonctionne pas',
                'prediction': 0.7
            }),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'reported')
        self.assertEqual(data['message'], 'Merci pour votre retour !')
    
    def test_report_missing_fields(self):
        """Tester la route /report avec des champs manquants."""
        response = self.app.post(
            '/report',
            data=json.dumps({
                'text': 'Test incomplet'
                # Champ 'prediction' manquant
            }),
            content_type='application/json'
        )
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
    
    @patch('app.preprocess_text')
    @patch('app.model')
    def test_predict_sentiment_function(self, mock_model, mock_preprocess):
        """Tester la fonction predict_sentiment directement."""
        # Configuration des mocks
        mock_preprocess.return_value = MagicMock()  # Simuler une séquence prétraitée
        mock_model.predict.return_value = [[0.75]]  # Simuler prédiction du modèle
        
        # Appeler la fonction
        result = predict_sentiment("Texte de test")
        
        # Vérifier que les mocks ont été appelés correctement
        mock_preprocess.assert_called_once_with("Texte de test")
        mock_model.predict.assert_called_once()
        
        # Vérifier le résultat
        self.assertEqual(result, 0.75)


if __name__ == '__main__':
    unittest.main()