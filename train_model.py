import pandas as pd
import joblib
import os

from utils import load_data, preprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

print("Loading dataset...")

df = load_data("dataset/fraud_dataset.csv", rows=50000)

df = preprocess(df)

print("Dataset shape:", df.shape)

# ================= TARGET =================
y = df["isFraud"]

# ================= FEATURES (MATCH APP INPUT) =================
features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest"
]

X = df[features]

# ================= SCALING =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= HANDLE IMBALANCE =================
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42
)

print("Training models...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        random_state=42
    )
}

results = {}

# ================= TRAIN MODELS =================
for name, model in models.items():

    print(f"\nTraining {name}")

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, pred)

    print("Accuracy:", acc)
    print("ROC AUC:", auc)
    print(classification_report(y_test, pred))

    results[name] = auc


# ================= BEST MODEL =================
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

print("\n===========================")
print("Best Model:", best_model_name)
print("Best ROC-AUC:", results[best_model_name])
print("===========================")

# ================= SAVE =================
os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

with open("models/best_model.txt", "w") as f:
    f.write(best_model_name)

print("Model saved successfully.")