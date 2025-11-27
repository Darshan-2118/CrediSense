import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a tiny dummy dataset
X = np.array([[5000, 200000, 0.4, 1, 0], [8000, 100000, 0.2, 0, 1]])
y = np.array([1, 0])

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

joblib.dump(model, 'dummy_model.joblib')
print('Saved dummy model to dummy_model.joblib')
