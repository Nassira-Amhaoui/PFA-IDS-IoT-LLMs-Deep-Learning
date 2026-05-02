from groq import Groq
import json
from dotenv import load_dotenv
import os

load_dotenv()

api_key 
GROQ_API_KEY = os.getenv("API_KEY")

client = Groq(api_key=GROQ_API_KEY)


def generer_alerte(type_attaque, confiance, details):

    prompt = f"""Tu es un expert en cybersécurité IoT.

Attaque détectée par notre système IDS :
- Type      : {type_attaque}
- Confiance : {confiance:.1%}
- Détails   : {json.dumps(details, ensure_ascii=False)}

Réponds uniquement en JSON :
{{
  "titre"           : "...",
  "gravite"         : "Faible|Moyen|Élevé|Critique",
  "description"     : "...",
  "recommandations" : ["...", "...", "..."]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0.3
    )

    texte = response.choices[0].message.content.strip()

    if "```json" in texte:
        texte = texte.split("```json")[1].split("```")[0]
    elif "```" in texte:
        texte = texte.split("```")[1].split("```")[0]

    return json.loads(texte)