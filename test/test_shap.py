import shap
import numpy as np
from sklearn.ensemble import RandomForestRegressor

X = np.random.rand(100, 5)
y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100)
model = RandomForestRegressor().fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X[0:1])
print("✅ SHAP works:", shap_values)