import os
import json
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement du fichier .env
load_dotenv()

# Récupération de la clé API de manière sécurisée
# On cherche "GROQ_API_KEY" ou "API_KEY" dans le fichier .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("API_KEY")

if not GROQ_API_KEY:
    # Optionnel : Une clé fictive pour éviter que l'importation ne crashe l'application entière
    GROQ_API_KEY = "gsk_fictive_key_for_testing_purposes"
    print("⚠️ Warning: Aucune clé API Groq trouvée dans le fichier .env. Utilisation d'une clé fictive.")

# Initialisation du client Groq
client = Groq(api_key=GROQ_API_KEY)


def generer_alerte(type_attaque, confiance, details):
    """
    Envoie les détails d'une attaque détectée à Llama 3.3 via Groq
    pour obtenir une analyse de criticité et des recommandations en JSON.
    """
    prompt = f"""Tu es un expert en cybersécurité IoT.

Attaque détectée par notre système IDS :
- Type      : {type_attaque}
- Confiance : {confiance:.1%}
- Détails   : {json.dumps(details, ensure_ascii=False)}

Réponds uniquement en JSON valide, sans texte avant ou après :
{{
  "titre"           : "...",
  "gravite"         : "Faible|Moyen|Élevé|Critique",
  "description"     : "...",
  "recommandations" : ["...", "...", "..."]
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.3
        )

        texte = response.choices[0].message.content.strip()

        # Nettoyage des balises Markdown de code block (```json ... ```)
        if "```json" in texte:
            texte = texte.split("```json")[1].split("```")[0]
        elif "```" in texte:
            texte = texte.split("```")[1].split("```")[0]

        return json.loads(texte.strip())

    except Exception as e:
        # En cas de problème (clé invalide, quota dépassé, erreur de parsing JSON),
        # on génère une réponse de secours pour ne pas bloquer l'IDS.
        print(f"❌ Erreur lors de la génération de l'alerte LLM : {e}")
        return {
            "titre": f"Alerte Automatique : {type_attaque}",
            "gravite": "Élevé",
            "description": f"Une attaque de type {type_attaque} a été détectée avec une confiance de {confiance:.1%}.",
            "recommandations": [
                "Isoler temporairement l'équipement IoT suspecté.",
                "Analyser les logs réseau locaux pour confirmer l'anomalie.",
                "Vérifier les règles du pare-feu."
            ]
        }