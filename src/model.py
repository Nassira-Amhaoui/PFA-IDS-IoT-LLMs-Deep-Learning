import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os



class GRU_LSTM(nn.Module):

    def __init__(self, n_features, n_classes):
        super(GRU_LSTM, self).__init__()

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
        out, _  = self.gru(x)
        out, _  = self.lstm(out)
        out     = out[:, -1, :]
        out     = self.bn(out)
        out     = self.dropout(out)
        out     = self.fc(out)
        return out



def construire_modele(n_features, n_classes):
    model = GRU_LSTM(n_features, n_classes)
    print("=== Architecture du modèle ===")
    print(model)
    return model



def entrainer(model, X_train, y_train, epochs=100, batch_size=1024):

    print("\n=== Entraînement PyTorch ===")

    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    model     = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.LongTensor(np.array(y_train)).to(device)

    val_size = int(len(X_tensor) * 0.1)
    X_val    = X_tensor[:val_size]
    y_val    = y_tensor[:val_size]
    X_tr     = X_tensor[val_size:]
    y_tr     = y_tensor[val_size:]

    dataset    = torch.utils.data.TensorDataset(X_tr, y_tr)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    history = {
        'loss': [], 'accuracy': [],
        'val_loss': [], 'val_accuracy': [],
        'lr': []
    }

    best_loss      = float('inf')
    patience_count = 0

    for epoch in range(epochs):

        model.train()
        total_loss = 0
        correct    = 0
        total      = 0

        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            predicted   = outputs.argmax(dim=1)
            correct    += (predicted == y_batch).sum().item()
            total      += y_batch.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        model.eval()
        with torch.no_grad():
            val_outputs  = model(X_val)
            val_loss     = criterion(val_outputs, y_val).item()
            val_pred     = val_outputs.argmax(dim=1)
            val_accuracy = (val_pred == y_val).float().mean().item()

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['lr'].append(current_lr)

        print(f"Epoch {epoch+1}/{epochs} "
              f"| Loss: {avg_loss:.4f} Acc: {accuracy:.4f} "
              f"| Val Loss: {val_loss:.4f} Val Acc: {val_accuracy:.4f} "
              f"| LR: {current_lr:.2e}")

        if val_loss < best_loss:
            best_loss      = val_loss
            patience_count = 0
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/gru_lstm_iot.pth')
            print(f"  ✔ Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= 10:
                print("Early stopping déclenché.")
                break

    print("\nModèle sauvegardé dans models/gru_lstm_iot.pth")
    return model, history



def evaluer(model, X_test, y_test, le):

    print("\n=== Évaluation ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()

    X_tensor = torch.FloatTensor(X_test).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        y_pred  = outputs.argmax(dim=1).cpu().numpy()

    classes_presentes = sorted(set(y_test) | set(y_pred))
    noms_classes      = le.inverse_transform(classes_presentes)

    os.makedirs('results', exist_ok=True)

   
    print(classification_report(
        y_test, y_pred,
        labels=classes_presentes,
        target_names=noms_classes,
        digits=3,
        zero_division=0
    ))

    with open('results/rapport.txt', 'w') as f:
        f.write(classification_report(
            y_test, y_pred,
            labels=classes_presentes,
            target_names=noms_classes,
            digits=3,
            zero_division=0
        ))
    print("Rapport sauvegardé dans results/rapport.txt")

  
    cm = confusion_matrix(y_test, y_pred, labels=classes_presentes)

    cm_normalise = cm.astype(float)
    row_sums     = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1      
    cm_normalise = cm_normalise / row_sums * 100

    fig_cm, ax = plt.subplots(
        figsize=(max(10, len(noms_classes) * 1.2),
                 max(8,  len(noms_classes) * 1.0))
    )

    sns.heatmap(
        cm_normalise,
        annot=True,
        fmt='.1f',
        cmap='Blues',
        xticklabels=noms_classes,
        yticklabels=noms_classes,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': 'Recall (%)'}
    )

    ax.set_title('Matrice de Confusion (% Recall par classe)', fontsize=14, pad=15)
    ax.set_xlabel('Classe Prédite',  fontsize=12)
    ax.set_ylabel('Classe Réelle',   fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig('results/matrice_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Matrice de confusion sauvegardée dans results/matrice_confusion.png")

    
    f1_par_classe = f1_score(
        y_test, y_pred,
        labels=classes_presentes,
        average=None,
        zero_division=0
    )

    accuracy_globale = accuracy_score(y_test, y_pred)
    f1_macro         = f1_score(y_test, y_pred, average='macro',    zero_division=0)
    f1_weighted      = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"\n📊 Métriques globales :")
    print(f"   Accuracy  : {accuracy_globale:.4f} ({accuracy_globale*100:.2f}%)")
    print(f"   F1 Macro  : {f1_macro:.4f}")
    print(f"   F1 Weighted : {f1_weighted:.4f}")

    indices_tries = np.argsort(f1_par_classe)[::-1]
    noms_tries    = [noms_classes[i] for i in indices_tries]
    f1_tries      = [f1_par_classe[i] for i in indices_tries]

    couleurs = ['#2ecc71' if v >= 0.9 else
                '#f39c12' if v >= 0.7 else
                '#e74c3c' for v in f1_tries]

    fig_f1, ax2 = plt.subplots(
        figsize=(max(10, len(noms_classes) * 0.9), 6)
    )

    bars = ax2.bar(noms_tries, f1_tries, color=couleurs, edgecolor='white', linewidth=0.8)

   
    for bar, val in zip(bars, f1_tries):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold'
        )


    ax2.axhline(y=f1_macro,    color='blue',   linestyle='--', linewidth=1.5,
                label=f'F1 Macro = {f1_macro:.3f}')
    ax2.axhline(y=f1_weighted, color='purple', linestyle=':',  linewidth=1.5,
                label=f'F1 Weighted = {f1_weighted:.3f}')
    ax2.axhline(y=0.9,         color='green',  linestyle='-',  linewidth=1.0,
                alpha=0.4, label='Seuil 0.90')

    ax2.set_title(
        f'F1-Score par classe  |  Accuracy globale : {accuracy_globale*100:.2f}%',
        fontsize=13, pad=12
    )
    ax2.set_xlabel('Classe', fontsize=11)
    ax2.set_ylabel('F1-Score', fontsize=11)
    ax2.set_ylim(0, 1.12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.4)

    from matplotlib.patches import Patch
    legende_couleurs = [
        Patch(color='#2ecc71', label='Excellent  (≥ 0.90)'),
        Patch(color='#f39c12', label='Acceptable (0.70–0.89)'),
        Patch(color='#e74c3c', label='Faible     (< 0.70)'),
    ]
    ax2.legend(handles=legende_couleurs + ax2.get_legend_handles_labels()[0][:3],
               fontsize=9, loc='upper right')

    plt.tight_layout()
    plt.savefig('results/f1_score.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Graphique F1-Score sauvegardé dans results/f1_score.png")

    return y_pred



def afficher_courbes(history):

    has_lr = 'lr' in history and len(history['lr']) > 0
    ncols  = 3 if has_lr else 2

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

    axes[0].plot(history['accuracy'],     label='Train',      color='steelblue')
    axes[0].plot(history['val_accuracy'], label='Validation', color='orange')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['loss'],     label='Train',      color='steelblue')
    axes[1].plot(history['val_loss'], label='Validation', color='orange')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    if has_lr:
        axes[2].plot(history['lr'], color='green', label='Learning Rate')
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True)

    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/courbes.png', dpi=150)
    plt.show()
    print("Courbes sauvegardées dans results/courbes.png")