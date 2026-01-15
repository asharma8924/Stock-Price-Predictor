print("RUNNING:", __file__)

from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[1]          # project root
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---- Dummy placeholders (replace with real pipeline once this runs) ----
def load_data():
    """
    TEMP: This is just to prove train.py runs.
    Replace this with your real feature-building code.
    """
    rng = np.random.RandomState(42)
    X = rng.randn(1000, 6)
    y = (X[:, 0] + 0.1 * rng.randn(1000) > 0).astype(int)
    return X, y

def split_data(X, y, train_frac=0.8):
    cut = int(len(X) * train_frac)
    return X[:cut], X[cut:], y[:cut], y[cut:]

def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=50,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)   # <- model is defined RIGHT HERE

    preds = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))

    out_path = MODELS_DIR / "random_forest.pkl"
    joblib.dump(model, out_path)
    print("Saved model to:", out_path)

    import joblib
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    MODELS_DIR = ROOT / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODELS_DIR / "rf_model.pkl")
    print("Model saved to:", MODELS_DIR / "rf_model.pkl")

