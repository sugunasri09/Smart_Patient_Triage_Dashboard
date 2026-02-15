import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# ---------- SYNTHETIC DATA ----------
np.random.seed(42)

n = 1000

age = np.random.randint(1, 90, n)
weight = np.random.randint(40, 120, n)
temp = np.random.uniform(96, 104, n)
hr = np.random.randint(60, 140, n)

# Risk generation logic (synthetic labels)
# Risk generation logic with slight noise
risk = []
for i in range(n):

    if temp[i] > 102 or hr[i] > 120:
        label = 2   # HIGH
    elif age[i] > 60:
        label = 1   # MEDIUM
    else:
        label = 0   # LOW

    # ---- ADD CONTROLLED NOISE (about 4%) ----
    if np.random.rand() < 0.04:
        label = np.random.choice([0,1,2])

    risk.append(label)


df = pd.DataFrame({
    "age": age,
    "weight": weight,
    "temp": temp,
    "hr": hr,
    "risk": risk
})
df.to_csv("synthetic_patient_data.csv", index=False)

X = df[["age","weight","temp","hr"]]
y = df["risk"]

# ---------- TRAIN MODEL ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Model trained successfully!")
print("Accuracy:", round(accuracy*100,2), "%")

# SAVE MODEL
joblib.dump(model, "model.pkl")
