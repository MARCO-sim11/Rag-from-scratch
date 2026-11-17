import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# documents d'exemple
sentences = [
    "Le projet Apollo 11 a permis aux premiers humains de marcher sur la Lune le 20 juillet 1969.",
    "La fusée utilisée pour la mission spaciale Apollo 11 était la Saturn V.",
    "L'assurance est un secteur hautement concurrrentiel et stratégique pour le secteur financier"
]

embedder = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3", 
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

embeddings = []

for sentence in sentences :
    response = embedder.embed_query(sentence)
    embeddings.append(response)
    print(f"Phrase : {sentence}")
    print(f"Embedding: [{response[0]:.4f}, {response[1]:.4f}, {response[2]:.4f}, ... ]")
    print(f"Longueur : {len(response)} numbers\n")

def dot_product(a,b):
    return sum(x * y for x, y in zip(a,b))

sim_1_2 = dot_product(embeddings[0], embeddings[1])
sim_1_3 = dot_product(embeddings[0], embeddings[2])

print("-" * 50)
print(f"Similitude phrase 1 vs phrase 2 -> {sim_1_2} ")
print(f"Similitude phrase 1 vs phrase 3 -> {sim_1_3} ")
print()
print("Remarque : les phrases similaires ont un score élevé, embeddings marche")

# python -m pip uninstall -y torch torchvision torchaudio
# Remove-Item -Recurse -Force "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch*"
# python -m pip install --no-cache-dir torch
# python -c "import torch; print(torch.__version__)"