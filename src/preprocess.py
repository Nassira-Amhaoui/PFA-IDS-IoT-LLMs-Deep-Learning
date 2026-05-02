import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib
import os


FAMILLES_CIBLES = ['DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Botnet']
LABEL_BENIN     = 'BenignTraffic'


MAPPING_EXPLICITE = {
    
    'DNS_Spoofing'          : 'Spoofing',
    'MITM-ArpSpoofing'      : 'Spoofing',

    'Backdoor_Malware'      : 'Botnet',
    'DictionaryBruteForce'  : 'Botnet',
    'CommandInjection'      : 'Botnet',
    'SqlInjection'          : 'Botnet',
    'Uploading_Attack'      : 'Botnet',
    'XSS'                   : 'Botnet',
    'BrowserHijacking'      : 'Botnet',
    'VulnerabilityScan'     : 'Botnet',
}


def mapper_famille(label):
    """
    Mappe un label détaillé vers sa famille.
    Priorité :
      1. BenignTraffic → conservé tel quel
      2. Mapping explicite → Spoofing / Botnet
      3. Préfixe → DDoS / DoS / Mirai / Recon
      4. Aucune correspondance → None (filtré)
    """
    if label == LABEL_BENIN:
        return LABEL_BENIN
    if label in MAPPING_EXPLICITE:
        return MAPPING_EXPLICITE[label]
    for famille in FAMILLES_CIBLES:
        if label.startswith(famille):
            return famille
    return None



def lire_dataset(chemin_csv, n_lignes=100_000):

    print(f"Lecture du fichier : {chemin_csv}")
    df = pd.read_csv(chemin_csv, nrows=n_lignes)
    print(f"Taille chargée     : {df.shape}")
    print(f"Colonnes           : {list(df.columns)}")
    print(f"Types de trafic    :\n{df['label'].value_counts()}\n")
    return df



def nettoyer(df):

    print("=== Nettoyage ===")
    print(f"Avant : {df.shape}")

    df = df.dropna()
    print(f"Après suppression NaN      : {df.shape}")

    df = df.drop_duplicates()
    print(f"Après suppression doublons : {df.shape}")

    cols_inutiles    = ['Covariance', 'Weight', 'Number']
    cols_a_supprimer = [c for c in cols_inutiles if c in df.columns]
    df = df.drop(columns=cols_a_supprimer, errors='ignore')
    print(f"Colonnes supprimées        : {cols_a_supprimer}")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"Après suppression infinis  : {df.shape}")

    print("Nettoyage terminé.\n")
    return df



def preparer(df):

    print("=== Préparation ===")

    df = df.copy()

   
    df['label'] = df['label'].apply(mapper_famille)
    df = df[df['label'].notna()].reset_index(drop=True)

    print(f"Classes après regroupement : {sorted(df['label'].unique())}")
    
    
    le = LabelEncoder()
    df['label_num'] = le.fit_transform(df['label'])

    print(f"\nClasses encodées ({len(le.classes_)}) :")
    for i, cls in enumerate(le.classes_):
        print(f"   {i} → {cls}")

    
    threshold          = 50
    comptage           = df['label_num'].value_counts()
    classes_valides    = comptage[comptage >= threshold].index
    classes_supprimees = comptage[comptage < threshold]

    if len(classes_supprimees) > 0:
        print(f"\nClasses supprimées (< {threshold} exemples) :")
        for cls, count in classes_supprimees.items():
            nom = le.inverse_transform([cls])[0]
            print(f"   {nom} → {count} exemples")

    df = df[df['label_num'].isin(classes_valides)].reset_index(drop=True)
    print(f"Lignes restantes : {df.shape[0]}")

  
    X = df.drop(columns=['label', 'label_num'])
    X = X.select_dtypes(include=[np.number])
    y = df['label_num'].values

    print(f"\nFeatures : {X.shape[1]}")

    
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nAvant équilibrage :")
    print(f"   Train : {X_train.shape[0]} lignes")
    print(f"   Test  : {X_test.shape[0]} lignes")

    
    print("\nDistribution AVANT équilibrage (train) :")
    comptage_avant = Counter(y_train)
    for cls, count in sorted(comptage_avant.items()):
        nom = le.inverse_transform([cls])[0]
        print(f"   {nom} : {count}")

    median_count = int(np.median(list(comptage_avant.values())))
    target_count = max(median_count, 500)
    print(f"\nNombre cible par classe : {target_count}")

    sampling_strategy = {
        cls: target_count
        for cls, count in comptage_avant.items()
        if count < target_count
    }

    if sampling_strategy:
        print("Application de SMOTE...")
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=42,
            k_neighbors=5
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("SMOTE terminé.")

   
    comptage_apres = Counter(y_train)
    indices_gardes = []

    for cls, count in comptage_apres.items():
        indices_cls = np.where(np.array(y_train) == cls)[0]
        if count > target_count * 3:
            indices_cls = np.random.choice(
                indices_cls,
                size=target_count * 3,
                replace=False
            )
        indices_gardes.extend(indices_cls.tolist())

    np.random.shuffle(indices_gardes)
    X_train = X_train[indices_gardes]
    y_train = np.array(y_train)[indices_gardes]

    print(f"\nDistribution APRÈS équilibrage (train) :")
    for cls, count in sorted(Counter(y_train).items()):
        nom = le.inverse_transform([cls])[0]
        print(f"   {nom} : {count}")

    print(f"\nTrain après équilibrage : {X_train.shape[0]} lignes")
    print(f"Test (inchangé)         : {X_test.shape[0]} lignes")

    X_train_rnn = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_rnn  = X_test.reshape(X_test.shape[0],  1, X_test.shape[1])

    print(f"Shape finale train : {X_train_rnn.shape}")
    print(f"Shape finale test  : {X_test_rnn.shape}\n")

    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le,     'models/label_encoder.pkl')
    print("Scaler et encodeur sauvegardés dans models/\n")

    return X_train_rnn, X_test_rnn, y_train, y_test, le, scaler