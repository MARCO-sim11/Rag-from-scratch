# Reads a pdf file, chunks it, embedds , and store it
import os
import fitz
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

# Charge la clé GROQ_API_KEY depuis votre fichier .env
load_dotenv()

# 1. Client Groq (servira UNIQUEMENT plus tard pour la génération de texte / chat)
# CORRECTION : L'URL de l'API est ://groq.com (et non https://groq.com)
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

PDF_PATH = "Global-Insurance-Report-2021.pdf" 
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        text = text.strip()
        if text : 
            pages.append((page_num + 1, text))
    doc.close()
    print(f"Texte extrait de {len(pages)} pages")
    return pages

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP ):
    """
    Découpage du texte en paquest identiques, avec overlap pour garder le contexte de chaque précédent
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start += chunk_size - overlap
    return chunks

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
) # plus adapté pour les documents lourds contrairement à SentenceTransformer
   

def embed_text(texts):
    """"
     Embbed multiple texts together
    """
    return embedder.embed_documents(texts)


def index_pdf(pdf_path):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    try : 
        chroma_client.delete_collection("pdf_rag")
    except :
        pass

    collection = chroma_client.create_collection("pdf_rag")
    
    pages = extract_text_from_pdf(pdf_path)

    all_chunks     = []
    all_metadata   = []
    all_ids        = []
    chunk_index    = 0 

    for page_num, page_text in pages : 

        for chunk in chunk_text(page_text):
            
            if len(chunk.strip()) < 50:
                continue
            
            all_chunks.append(chunk)
            all_metadata.append({
                "source": os.path.basename(pdf_path),
                "page"  : page_num,
                "chunk" : chunk_index
            })
            all_ids.append(f"chunk_{chunk_index}")
            chunk_index += 1

    BATCH_SIZE = 100
    all_embeddings = []

    for i in range (0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i: i + BATCH_SIZE]
        embeddings = embed_text(batch)
        all_embeddings.extend(embeddings)
        print(f"  Embdedded batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadata,
        ids=all_ids
    )

    print (f"\n Done ! Indexed {len(all_chunks)} chunks from {pdf_path}")

    return collection

def ask(collection, question, n_results=3):
    question_embedding = embed_text([question])[0]

    results = collection.query(
        query_embeddings = [question_embedding],
        n_results = n_results,
        include = ["documents", "metadatas", "distances"]
    )

    chunks    =  results["documents"][0]
    metadatas =  results["metadatas"][0]
    distances =  results["distances"][0]

    print(f"\nQuestion : {question}")
    print("\nRetrieved chunks :")
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        print(f"    [Page {meta['page']}  |  score: {dist:.3f}] {chunk[:100]}...")

    context = "\n\n---\n\n".join(chunks)
    prompt = f""""Réponds à la question en utilisant UNIQUEMENT le contexte ci-dessous.
    Toujours mentionner de quelle page la réponse vient. 
    Si la réponse n'apparait pas dans le contexte, dire " je ne connais pas".

    Contexte : 
    {context}

    Question : {question}
    Réponse : """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role":"user", "content": prompt}]
        )

    print(f"\nAnswer: {response.choices[0].message.content}")


if __name__=="__main__":
    collection = index_pdf(PDF_PATH)
    ask(collection, "Why will the risk landscape change after Covid-19 ?")
    ask(collection, "what are the expected trends in the insurance sector for 2021 and why ?")