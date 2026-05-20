import torch
import numpy as np
import joblib
from model import HybridModel
from alert import generer_alerte


def charger_modele(n_features, n_classes):
    model = GRU_LSTM(n_features, n_classes)
    model.load_state_dict(torch.load('models/gru_lstm_iot.pth'))
    model.eval()
    scaler = joblib.load('models/scaler.pkl')
    le     = joblib.load('models/label_encoder.pkl')
    return model, scaler, le


def analyser_paquet(features, model, scaler, le, seuil=0.6):

    # X_test est déjà normalisé et en shape (1, 43)
    # On enlève la dimension timestep si présente
    if features.ndim == 2:
        # shape (1, 43) → (43,)
        features_flat = features[0]
    elif features.ndim == 1:
        # shape (43,) → déjà bon
        features_flat = features
    else:
        features_flat = features.flatten()

    # Reshape pour le modèle : (1, 1, 43)
    features_tensor = torch.FloatTensor(
        features_flat.reshape(1, 1, -1)
    )

    # Prédiction
    model.eval()
    with torch.no_grad():
        output     = model(features_tensor)
        proba      = torch.softmax(output, dim=1)
        classe_idx = proba.argmax().item()
        confiance  = proba.max().item()

    type_trafic = le.inverse_transform([classe_idx])[0]

    print(f"Résultat  : {type_trafic}")
    print(f"Confiance : {confiance:.1%}")

    if type_trafic != 'BenignTraffic' and confiance > seuil:
        print("ATTAQUE DÉTECTÉE — génération alerte LLM...")

        details = {
            "type"      : type_trafic,
            "confiance" : f"{confiance:.1%}"
        }

        alerte = generer_alerte(type_trafic, confiance, details)

        print(f"\nALERTE      : {alerte['titre']}")
        print(f"Gravité     : {alerte['gravite']}")
        print(f"Description : {alerte['description']}")
        print("Recommandations :")
        for r in alerte['recommandations']:
            print(f"  - {r}")

        return alerte

    else:
        print("Trafic normal — aucune alerte.")
        return None