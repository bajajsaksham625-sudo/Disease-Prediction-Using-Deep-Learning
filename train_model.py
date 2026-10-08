import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

DATASET = "dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "disease_model.keras")
LABEL_PATH = os.path.join(MODEL_DIR, "labels.json")

df = pd.read_csv(DATASET)
symptom_cols = [c for c in df.columns if c != "disease"]

X = df[symptom_cols].astype("float32").values
encoder = LabelEncoder()
y_int = encoder.fit_transform(df["disease"])
y = keras.utils.to_categorical(y_int, num_classes=len(encoder.classes_))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y_int
)

model = keras.Sequential([
    layers.Input(shape=(len(symptom_cols),)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(32, activation="relu"),
    layers.Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training model...")
model.fit(
    X_train, y_train,
    validation_split=0.20,
    epochs=80,
    batch_size=16,
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {accuracy:.2%}")

os.makedirs(MODEL_DIR, exist_ok=True)
model.save(MODEL_PATH)

with open(LABEL_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "symptoms": symptom_cols,
        "classes": encoder.classes_.tolist()
    }, f, indent=2)

print(f"Saved model to: {MODEL_PATH}")
print(f"Saved metadata to: {LABEL_PATH}")
