import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer

# Charge la clé GROQ_API_KEY depuis votre fichier .env
load_dotenv()

# 1. Client Groq (servira UNIQUEMENT plus tard pour la génération de texte / chat)
# CORRECTION : L'URL de l'API est ://groq.com (et non https://groq.com)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

# 2. Modèle d'embeddings 100% LOCAL et gratuit (remplace text-embedding-3-small d'OpenAI)
# Téléchargé une seule fois sur votre PC, tourne ensuite hors-ligne sans clé API
embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") 

documents = [
    "La politique de retour permet aux clients de retourner les articles dans un délai de 30 jours suivant l'achat.",
    "Tous les retours doivent inclure le reçu original ainsi que l'emballage d'origine.",
    "Les remboursements sont traités sous 5 à 7 jours ouvrés après réception de l'article.",
    "Les articles endommagés ou utilisés ne peuvent pas être retournés, sauf s'ils sont défectueux.",
    "Pour initier un retour, rendez-vous sur notre site web et remplissez le formulaire de demande de retour.",
]


# Initialisation de la base vectorielle locale ChromaDB en mémoire
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="my_first_rag")

print("Embedding and storing documents...")

for i, doc in enumerate(documents):
    # CORRECTION : On génère le vecteur localement au lieu d'appeler client.embeddings.create()
    # .tolist() convertit le tableau NumPy en liste de floats Python (format attendu par ChromaDB)
    embedding = embed_model.encode(doc).tolist()

    # Ajout du document, de son vecteur et de son ID unique dans ChromaDB
    collection.add(
        documents=[doc],
        embeddings=[embedding],
        ids=[f"doc_{i}"]
    )

print(f"Stored {len(documents)} documents in ChromaDB \n")

# Une question qu'on numérise
question = "De combien de temps je dispose pour retourner un article ?"
question_embedding = embed_model.encode(question).tolist()

# On cherche les 2 vecteurs les plus proche dans la collection
results = collection.query(
    query_embeddings = [question_embedding],
    n_results = 2
)


# Le résultat
retrieved_chunks = results["documents"][0] # liste de correspondant
print("Retrieved_chunks : ")
for chunk in retrieved_chunks:
    print(f" -> {chunk}")
print()

context = "\n".join(retrieved_chunks)

prompt = f"""Tu es un agent utile. Répond à la question en utilisant UNIQUEMENT le contexte qui suit.
Si la réponse ne figure pas dans le contexte, réponds "Je ne sais pas."

Contexte :
{context}

Question : {question}
Réponse : """

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": prompt}]
)

answer = response.choices[0].message.content
print(f"Question: {question}")
print(f"Answer: {answer}")