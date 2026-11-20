import os
import fitz
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

app = FastAPI(title="RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_or_create_collection():
    try :
        return chroma_client.get_collection("rag_collection")
    except:
        return chroma_client.create_collection("rag_collection")


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text : 
            pages.append((i+1, text))
    doc.close()
    return pages

def chunk_text(text):
    """
    Découpage du texte en paquest identiques, avec overlap pour garder le contexte de chaque précédent
    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

# Building local embedder
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
) # plus adapté pour les documents lourds contrairement à SentenceTransformer
   

def embed_texts(texts):
    """"
     Embbed multiple texts together
    """
    return embedder.embed_documents(texts)

@app.post("/ingest")
async def ingest(file:UploadFile=File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    collection = get_or_create_collection()

    pages = extract_text(tmp_path)
    all_chunks    = []
    all_metadata  = []
    all_ids       = []
    chunk_index   = collection.count()

    for page_num, page_text in pages : 
        for j, chunk in enumerate(chunk_text(page_text)):
            if len(chunk.strip()) < 50:
                continue
            all_chunks.append(chunk)
            all_metadata.append({
                "source":file.filename,
                "page": page_num
            })
            all_ids.append(f"{file.filename}_p{page_num}_c{j}") #all_ids.append(f"chunk_{chunk_index}")
            # chunk_index += 1

    
    all_embeddings = []
    for i in range (0, len(all_chunks), 100):
        batch = all_chunks[i : i + 100]
        all_embeddings.extend(embed_texts(batch))

    # Store in ChromaDB
    collection.upsert(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadata,
        ids=all_ids,
    )

    os.unlink(tmp_path)

    return {
        "message":f"Ingested {file.filename} successfully",
        "chunks_added": len(all_chunks),
        "total_chunks": collection.count(),
    }



class AskRequest(BaseModel):
    question : str
    n_result : int = 3

@app.post("/ask")
async def ask(request: AskRequest):

    collection = get_or_create_collection()

    if collection.count() == 0:
        return {"error":"No Documents ingested yet"}
    
    question_embedding = embedder.embed_query(request.question) # question_embedding = embed_texts(request.question) #question_embedding = embed_texts([request.question[0]])


    # find chunks similar to the question
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=request.n_result,
        include=["documents", "metadatas", "distances"]
    )

    chunks    = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []

    sources = []

    for chunk, meta, dist in zip(chunks, metadatas, distances):
        context_parts.append(chunk)
        sources.append({
            "source": meta["source"],
            "page"  : meta["page"],
            "score" : round(dist, 4),
        })

    context = "\n\n---\n\n".join(context_parts)

    prompt = f""" Tu es un assistant utile et percutant, très analytique. Réponds à la question en utilisant UNIQUEMENT le contexte ci-dessous.
    Sois précis et concis. Mentionne toujours le document source ainsi que la page si besoin.
    Si la réponse ne se trouve pas dans le contexte, réponds "Les documents fournis ne permettent pas de répondre" ou autre réponse similaire.
    
    Contexte : 
    {context}
    
    Question : {request.question}
    Réponse"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role":"user", "content": prompt}]
        )

    return {
        "question": request.question,
        "answer": response.choices[0].message.content,
        "sources": sources,
    }

@app.get("/")
def root():
    collection = get_or_create_collection()
    return{
        "status":   "running",
        "total_chunks": collection.count(),
    }
