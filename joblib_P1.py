import joblib
import numpy as np
model = joblib.load("P1_model.joblib")
X_new = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
predictions = model.predict(X_new)

print("Maintenance Steps:")
print(predictions,  "\n")