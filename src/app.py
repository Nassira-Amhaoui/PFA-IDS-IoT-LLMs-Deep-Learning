import streamlit as st
import torch
import numpy as np
import joblib
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GRU_LSTM
from detector import charger_modele, analyser_paquet

st.set_page_config(page_title="IoT IDS", layout="wide")
st.title("Système de Détection d'Intrusions IoT")
st.markdown("---")


n_features = 43
n_classes  = 34

model, scaler, le = charger_modele(n_features, n_classes)


col1, col2, col3 = st.columns(3)
col1.metric("Accuracy du modèle", "82.3%")
col2.metric("Classes détectées",  "32 types")
col3.metric("Dataset",            "100 000 lignes")

st.markdown("---")


st.subheader("Courbes d'apprentissage")
st.image("results/courbes.png")

st.markdown("---")

st.subheader("Tester une détection")
seuil = st.slider("Seuil de confiance", 0.1, 0.9, 0.4)

if st.button("Analyser un exemple aléatoire"):
    scaler_raw = joblib.load('models/scaler.pkl')
    features   = np.random.randn(n_features)

    features_tensor = torch.FloatTensor(
        features.reshape(1, 1, -1)
    )

    model.eval()
    with torch.no_grad():
        output     = model(features_tensor)
        proba      = torch.softmax(output, dim=1)
        classe_idx = proba.argmax().item()
        confiance  = proba.max().item()

    type_trafic = le.inverse_transform([classe_idx])[0]

    if type_trafic != 'BenignTraffic' and confiance > seuil:
        st.error(f"ATTAQUE : {type_trafic} ({confiance:.1%})")
    else:
        st.success(f"Trafic normal : {type_trafic} ({confiance:.1%})")