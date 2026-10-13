from contextlib import asynccontextmanager
from pathlib import Path
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / 'model' / 'churn_model.joblib'
FEATURES = ['tenure_months', 'monthly_charge', 'support_calls']

REQUESTS = Counter('predict_requests_total', 'Prediction requests', ['outcome'])
LATENCY = Histogram('predict_latency_seconds', 'Time spent predicting')
CHURN_PROB = Histogram('predict_churn_probability', 'Predicted churn probability',
                        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

state = {'model': None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # सर्वर चालू होते ही मॉडल को लोड करता है
    state['model'] = joblib.load(MODEL_PATH)
    yield

app = FastAPI(title='Churn prediction API', lifespan=lifespan)

class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charge: float = Field(ge=0, le=1000)
    support_calls: int = Field(ge=0, le=100)

@app.get('/healthz')
def healthz():
    return {'status': 'ok'}

@app.get('/readyz')
def readyz():
    if state['model'] is None:
        raise HTTPException(status_code=503, detail='model not loaded')
    return {'status': 'ready'}

@app.post('/predict')
def predict(customer: Customer):
    if state['model'] is None:
        REQUESTS.labels(outcome='error').inc()
        raise HTTPException(status_code=503, detail='model not loaded')
    
    start = time.perf_counter()
    row = pd.DataFrame([customer.model_dump()])[FEATURES]
    prob = float(state['model'].predict_proba(row)[0][1])
    
    LATENCY.observe(time.perf_counter() - start)
    CHURN_PROB.observe(prob)
    REQUESTS.labels(outcome='ok').inc()
    
    return {'churn_probability': round(prob, 4), 'will_churn': prob >= 0.5}

@app.get('/metrics')
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

