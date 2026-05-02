import sys
import os
import glob
import pandas as pd
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import nettoyer, preparer, mapper_famille
from model import construire_modele, entrainer, evaluer, afficher_courbes
from detector import charger_modele, analyser_paquet



def lire_tous_fichiers(
    dossier,
    chunksize=50000,
    output_cache="data/cache_filtre.csv",
    max_par_famille=150000     
):
    """
    - Lit les CSV par chunks
    - Filtre immédiatement par famille d'attaque
    - Limite à max_par_famille lignes par classe → évite disque plein
    - Sauvegarde sur disque chunk par chunk → évite RAM pleine
    - Si cache existe déjà → recharge directement
    """

  
    if os.path.exists(output_cache):
        print(f" Cache trouvé → chargement direct de '{output_cache}'")
        df = pd.read_csv(output_cache, low_memory=False)
        print(f"   {df.shape[0]:,} lignes  |  {df.shape[1]} colonnes\n")
        return df

    fichiers = sorted(glob.glob(os.path.join(dossier, "*.csv")))
    if not fichiers:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans : {dossier}")

    print(f" {len(fichiers)} fichiers CSV trouvés")
    print(f" Limite : {max_par_famille:,} lignes max par famille")
    print("-" * 50)

    
    compteur = defaultdict(int)

   
    familles_cibles = ['DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Botnet', 'BenignTraffic']

    cache_dir = os.path.dirname(output_cache)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    premiere_ecriture = True
    total_lignes      = 0
    toutes_completes  = False

    for i, f in enumerate(fichiers, 1):
        nom = os.path.basename(f)

      
        if all(compteur[fam] >= max_par_famille for fam in familles_cibles):
            print(f"\n🏁 Toutes les familles ont atteint {max_par_famille:,} lignes → arrêt anticipé")
            toutes_completes = True
            break

        lignes_fichier = 0

        try:
            for chunk in pd.read_csv(f, chunksize=chunksize, low_memory=False):

                chunk = chunk.copy()
                chunk['famille'] = chunk['label'].apply(mapper_famille)
                chunk = chunk[chunk['famille'].notna()]

                
                chunks_filtres = []
                for fam in chunk['famille'].unique():
                    reste = max_par_famille - compteur[fam]
                    if reste <= 0:
                        continue
                    sous = chunk[chunk['famille'] == fam].head(reste)
                    compteur[fam] += len(sous)
                    chunks_filtres.append(sous)

                if not chunks_filtres:
                    continue

                chunk_final = pd.concat(chunks_filtres).drop(columns=['famille'])

                if len(chunk_final) > 0:
                    chunk_final.to_csv(
                        output_cache,
                        mode='w' if premiere_ecriture else 'a',
                        header=premiere_ecriture,
                        index=False
                    )
                    premiere_ecriture = False
                    lignes_fichier   += len(chunk_final)

        except Exception as e:
            print(f"    Fichier ignoré ({nom}) : {e}")
            continue

        total_lignes += lignes_fichier
        print(f"  [{i:02d}/{len(fichiers)}] {nom}  →  {lignes_fichier:,} lignes conservées")

  
    print("-" * 50)
    print("Lignes collectées par famille :")
    for fam in familles_cibles:
        print(f"   {fam:<15} : {compteur[fam]:,}")

    print(f"\n Cache sauvegardé : '{output_cache}'")
    print(f"   Total : {total_lignes:,} lignes\n")

    
    print("Chargement du cache en mémoire...")
    dfs = []
    for chunk in pd.read_csv(output_cache, chunksize=chunksize, low_memory=False):
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    print(f" {df.shape[0]:,} lignes  |  {df.shape[1]} colonnes\n")

    return df


if __name__ == "__main__":

    DATASET_PATH = "data/CICIOT2023"

    
    df = lire_tous_fichiers(
        DATASET_PATH,
        chunksize=50000,
        output_cache="data/cache_filtre.csv",
        max_par_famille=150000      
    )


    df = nettoyer(df)

   
    X_train, X_test, y_train, y_test, le, scaler = preparer(df)

  
    n_features = X_train.shape[2]
    n_classes  = len(le.classes_)
    print(f"n_features = {n_features}  |  n_classes = {n_classes}")

    model = construire_modele(n_features, n_classes)

    model, history = entrainer(model, X_train, y_train, epochs=50)

    afficher_courbes(history)

    y_pred = evaluer(model, X_test, y_test, le)

    print("\n=== Test détection ===")
    model2, scaler2, le2 = charger_modele(n_features, n_classes)

    for i in range(5):
        print(f"\n--- Exemple {i+1} ---")
        analyser_paquet(X_test[i], model2, scaler2, le2)

    print("\n Terminé. Résultats dans results/")