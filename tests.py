import json
import unittest
from app import app

class TestSentimentAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_health_endpoint(self):
        """Test que le point de terminaison /health répond correctement"""
        response = self.app.get('/health')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'healthy')
    
    def test_predict_endpoint_positive(self):
        """Test la prédiction avec un tweet positif"""
        response = self.app.post('/predict',
                                data=json.dumps({'text': 'I love flying with Air France!'}),
                                content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('sentiment', data)
        self.assertIn('sentiment_label', data)
        # Vous ne pouvez pas tester la valeur exacte car elle dépend du modèle
        self.assertIsInstance(data['sentiment'], float)
    
    def test_predict_endpoint_negative(self):
        """Test la prédiction avec un tweet négatif"""
        response = self.app.post('/predict',
                                data=json.dumps({'text': 'Worst flight ever, never flying with Air France again!'}),
                                content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn('sentiment', data)
        self.assertIsInstance(data['sentiment'], float)
    
    def test_predict_endpoint_missing_text(self):
        """Test le comportement quand le champ 'text' est manquant"""
        response = self.app.post('/predict',
                                data=json.dumps({}),
                                content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
    
    def test_report_endpoint(self):
        """Test le signalement d'une mauvaise prédiction"""
        response = self.app.post('/report',
                                data=json.dumps({'text': 'This flight was amazing!', 'prediction': 0.2}),
                                content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'reported')

if __name__ == '__main__':
    unittest.main(verbosity=2)