# Disease Prediction Using Deep Learning

A simple educational Deep Learning project that predicts a disease label from a set of symptoms.

## Important
This is a **college/educational demonstration only**. It is not a medical diagnostic tool and should not be used for real medical decisions.

## Technology
- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Project structure
```text
disease_prediction_dl/
├── app.py
├── train_model.py
├── dataset.csv
├── requirements.txt
├── README.md
└── models/
    └── disease_model.keras   # generated after training
```

## How to run

### 1. Install Python
Python 3.10 or newer is recommended.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```

This creates `models/disease_model.keras` and preprocessing files.

### 4. Start the web app
```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## How it works
1. Symptoms are represented as binary input features.
2. A small Keras neural network learns patterns in the training dataset.
3. The model outputs probabilities for the disease classes.
4. The Streamlit interface displays the highest-probability class.

The included dataset is intentionally small and synthetic/educational. For an academic submission, you can replace it with a properly sourced public dataset and cite its source.
