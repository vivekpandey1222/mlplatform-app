import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = ['tenure_months', 'monthly_charge', 'support_calls']

# 1. नकली डेटा जनरेट करना
rng = np.random.default_rng(42)
n = 3000
df = pd.DataFrame({
    'tenure_months': rng.integers(1, 72, n),
    'monthly_charge': rng.uniform(20, 120, n).round(2),
    'support_calls': rng.integers(0, 10, n),
})
score = -0.04 * df.tenure_months + 0.02 * df.monthly_charge + 0.35 * df.support_calls
prob = 1 / (1 + np.exp(-(score - 3.0)))
df['churned'] = (rng.random(n) < prob).astype(int)

# 2. टेस्ट करने के लिए 20% डेटा अलग रखना
X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df['churned'], test_size=0.2, random_state=42,
    stratify=df['churned'])

# 3. मॉडल को ट्रेन करना
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. मॉडल की एक्यूरेसी नापना
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
print('accuracy:', round(accuracy_score(y_test, pred), 3))
print('baseline (always say no churn):', round(1 - y_test.mean(), 3))
print('roc_auc:', round(roc_auc_score(y_test, proba), 3))

# 5. मॉडल फाइल को सेव करना
joblib.dump(model, 'model/churn_model.joblib')
df[FEATURES].describe().to_csv('model/train_stats.csv')
print('saved model/churn_model.joblib')

