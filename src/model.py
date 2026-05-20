import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from matplotlib.patches import Patch

# Détermination automatique du Device (GPU si disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Gestion dynamique des chemins absolus
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_LE = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "label_encoder.pkl"))
PATH_SCALER = os.path.abspath(os.path.join(BASE_DIR, "..", "models", "scaler.pkl"))


# ============================================================
# ARCHITECTURE DU MODÈLE HYBRIDE GRU + LSTM
# ============================================================
class HybridModel(nn.Module):
    def __init__(self, n_features, n_classes):
        super(HybridModel, self).__init__()
        self.gru  = nn.GRU(n_features, 256, batch_first=True)
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        self.bn      = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out, _ = self.lstm(out)
        out    = out[:, -1, :]   # Dernier état temporel
        out    = self.bn(out)
        out    = self.dropout(out)
        return self.fc(out)


# ============================================================
# FONCTIONS REQUISES PAR MAIN.PY
# ============================================================

def construire_modele(n_features, n_classes):
    """Initialise le modèle hybride et l'envoie sur le device (CPU/GPU)"""
    model = HybridModel(n_features, n_classes).to(device)
    print("\n=== Architecture du modèle ===")
    print(model)
    return model


def entrainer(model, X_train_res, y_train_res, epochs=50):
    """
    Gère la création des dataloaders (avec split de validation),
    la boucle d'entraînement, le learning rate scheduler et l'early stopping.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )

    # Reformatage 3D pour l'entrée RNN -> (samples, seq_len=1, features)
    if len(X_train_res.shape) == 2:
        X_train_res = X_train_res.reshape(X_train_res.shape[0], 1, X_train_res.shape[1])

    # Split interne 90/10 pour la validation
    val_size = int(len(X_train_res) * 0.1)
    X_val_rnn = X_train_res[:val_size]
    y_val_arr = y_train_res[:val_size]
    X_tr_rnn  = X_train_res[val_size:]
    y_tr_arr  = y_train_res[val_size:]

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr_rnn), torch.LongTensor(y_tr_arr)),
        batch_size=1024, shuffle=True
    )

    X_val_tensor = torch.FloatTensor(X_val_rnn).to(device)
    y_val_tensor = torch.LongTensor(y_val_arr).to(device)

    history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_val_loss = float('inf')
    patience_count = 0
    
    os.makedirs(os.path.join(BASE_DIR, '..', 'models'), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, '..', 'results'), exist_ok=True)

    print(f"\nDébut de l'entraînement sur {device} pour {epochs} époques...\n")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss, correct = 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == y_batch).sum().item()

        # Métriques d'entraînement
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / len(y_tr_arr)

        # Métriques de validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_acc = (val_outputs.argmax(1) == y_val_tensor).float().mean().item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        duration = time.time() - start_time

        # Enregistrement de l'historique
        history['loss'].append(epoch_loss)
        history['acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch+1:02d}/{epochs} "
              f"| Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} "
              f"| Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f} "
              f"| LR: {current_lr:.2e} | {duration:.1f}s")

        # Sauvegarde du meilleur modèle + Early Stopping
        model_path = os.path.join(BASE_DIR, '..', 'models', 'hybrid_iot.pth')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✔ Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= 10:
                print("Early stopping déclenché.")
                break

    return model, history


def afficher_courbes(history):
    """Génère et sauvegarde les courbes d'apprentissage"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    axes[0].plot(history['acc'], label='Train', color='steelblue')
    axes[0].plot(history['val_acc'], label='Validation', color='orange')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['loss'], label='Train', color='steelblue')
    axes[1].plot(history['val_loss'], label='Validation', color='orange')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(history['lr'], color='green', label='Learning Rate')
    axes[2].set_title('Learning Rate')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    output_img = os.path.join(BASE_DIR, '..', 'results', 'courbes.png')
    plt.savefig(output_img, dpi=150)
    plt.show()
    print(f"Courbes sauvegardées dans {output_img}")


def evaluer(model, X_test, y_test, le):
    """Évalue le modèle sur le jeu de test et génère les rapports et matrices"""
    # Rechargement des poids optimaux
    model_path = os.path.join(BASE_DIR, '..', 'models', 'hybrid_iot.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if len(X_test.shape) == 2:
        X_test_rnn = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    else:
        X_test_rnn = X_test

    X_test_tensor = torch.FloatTensor(X_test_rnn).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    classes_presentes = sorted(set(y_test) | set(y_pred))
    noms_classes = le.inverse_transform(classes_presentes)

    # Rapport textuel
    rapport = classification_report(
        y_test, y_pred, labels=classes_presentes, target_names=noms_classes, digits=3, zero_division=0
    )
    print("\n=== Rapport de classification ===")
    print(rapport)

    rapport_path = os.path.join(BASE_DIR, '..', 'results', 'rapport.txt')
    with open(rapport_path, 'w') as f:
        f.write(rapport)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=classes_presentes)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums * 100

    fig_cm, ax = plt.subplots(figsize=(max(10, len(noms_classes) * 1.2), max(8, len(noms_classes) * 1.0)))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', xticklabels=noms_classes, yticklabels=noms_classes, ax=ax, linewidths=0.5, cbar_kws={'label': 'Recall (%)'})
    ax.set_title('Matrice de Confusion (% Recall par classe)', fontsize=14, pad=15)
    ax.set_xlabel('Classe Prédite', fontsize=12)
    ax.set_ylabel('Classe Réelle', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    cm_path = os.path.join(BASE_DIR, '..', 'results', 'matrice_confusion.png')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.show()

    # Métriques globales & Graphique F1-Score
    accuracy_globale = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_par_classe = f1_score(y_test, y_pred, labels=classes_presentes, average=None, zero_division=0)

    print(f"\n📊 Métriques globales :")
    print(f"   Accuracy    : {accuracy_globale*100:.2f}%")
    print(f"   F1 Macro    : {f1_macro:.4f}")
    print(f"   F1 Weighted : {f1_weighted:.4f}")

    # Plot F1-Scores
    indices_tries = np.argsort(f1_par_classe)[::-1]
    noms_tries = [noms_classes[i] for i in indices_tries]
    f1_tries = [f1_par_classe[i] for i in indices_tries]
    couleurs = ['#2ecc71' if v >= 0.9 else '#f39c12' if v >= 0.7 else '#e74c3c' for v in f1_tries]

    fig_f1, ax2 = plt.subplots(figsize=(max(10, len(noms_classes) * 0.9), 6))
    bars = ax2.bar(noms_tries, f1_tries, color=couleurs, edgecolor='white', linewidth=0.8)

    for bar, val in zip(bars, f1_tries):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.axhline(y=f1_macro, color='blue', linestyle='--', linewidth=1.5, label=f'F1 Macro = {f1_macro:.3f}')
    ax2.axhline(y=f1_weighted, color='purple', linestyle=':', linewidth=1.5, label=f'F1 Weighted = {f1_weighted:.3f}')
    
    legende_couleurs = [
        Patch(color='#2ecc71', label='Excellent  (≥ 0.90)'),
        Patch(color='#f39c12', label='Acceptable (0.70–0.89)'),
        Patch(color='#e74c3c', label='Faible     (< 0.70)'),
    ]
    ax2.legend(handles=legende_couleurs + ax2.get_legend_handles_labels()[0][:2], fontsize=9, loc='upper right')
    ax2.set_title(f'F1-Score par classe  |  Accuracy : {accuracy_globale*100:.2f}%', fontsize=13, pad=12)
    ax2.set_ylim(0, 1.12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.4)

    f1_path = os.path.join(BASE_DIR, '..', 'results', 'f1_score.png')
    plt.tight_layout()
    plt.savefig(f1_path, dpi=150, bbox_inches='tight')
    plt.show()

    return y_pred


# ============================================================
# EXECUTION STANDALONE (SI TRACE DIRECT)
# ============================================================
if __name__ == "__main__":
    # Ce bloc ne s'exécutera QUE si vous lancez `python src/model.py` directement.
    # Utile pour tester le script indépendamment du main.py
    print("Exécution du script en mode autonome...")
    
    PATH_DATA = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "cache_filtre.csv"))
    if not os.path.exists(PATH_DATA):
        print(f"Erreur : Le fichier cache '{PATH_DATA}' n'existe pas. Lancez main.py d'abord.")
    else:
        le = joblib.load(PATH_LE)
        scaler = joblib.load(PATH_SCALER)
        df = pd.read_csv(PATH_DATA)

        X = df.drop(columns=['label'])
        y = le.transform(df['label'])
        X_scaled = scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

        print("Application de SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        model = construire_modele(X_train_res.shape[1], len(le.classes_))
        model, history = entrainer(model, X_train_res, y_train_res, epochs=5)
        afficher_courbes(history)
        evaluer(model, X_test, y_test, le)