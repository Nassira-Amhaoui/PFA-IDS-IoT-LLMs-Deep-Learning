import glob
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

fichiers = sorted(glob.glob("data/CICIOT2023/*.csv"))[:20]  

print(f"Lecture de {len(fichiers)} fichiers...")

labels = set()
for f in fichiers:
    df = pd.read_csv(f, usecols=['label'], low_memory=False)
    labels.update(df['label'].unique())

print(f"\n=== TOUS LES LABELS UNIQUES ({len(labels)}) ===")
for l in sorted(labels):
    print(f"  {l}")