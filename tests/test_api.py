from fastapi.testclient import TestClient
from app.main import app

def test_health_ready_and_predict():
    with TestClient(app) as client:
        assert client.get('/healthz').status_code == 200
        assert client.get('/readyz').status_code == 200
        r = client.post('/predict', json={'tenure_months': 2, 'monthly_charge': 110, 'support_calls': 8})
        assert r.status_code == 200
        assert 0 <= r.json()['churn_probability'] <= 1

def test_rejects_bad_input():
    with TestClient(app) as client:
        # गलत डेटा (जैसे माइनस में महीने) भेजने पर एरर आना चाहिए (422)
        r = client.post('/predict', json={'tenure_months': -5, 'monthly_charge': 50, 'support_calls': 1})
        assert r.status_code == 422

