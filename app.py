import os
import time
import faiss
import numpy as np
import requests
from PyPDF2 import PdfReader
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import tiktoken
from io import BytesIO

# --- CONFIG ---
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)
encoding = tiktoken.get_encoding("cl100k_base")

app = FastAPI(title="Zoning Plan Chatbot")

# Enable CORS for your Bubble app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your Bubble domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REQUEST/RESPONSE MODELS ---
class QueryIn(BaseModel):
    query: str
    planid: str
    gmina: str | None = None
    parcelid: str | None = None
    pdfurl: str

class QueryOut(BaseModel):
    answer: str
    planid: str

# --- UTIL FUNCTIONS ---

def extract_text_from_pdf_url(pdf_url: str) -> str:
    """Fetch and extract text directly from a PDF URL (no saving to disk)."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/118.0 Safari/537.36"
            ),
            "Referer": "https://brwinow.e-mapa.net/",
            "Accept": "application/pdf",
        }
        r = requests.get(pdf_url, headers=headers, timeout=30)
        r.raise_for_status()

        pdf_file = BytesIO(r.content)
        reader = PdfReader(pdf_file)

        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text.strip()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {e}")

def chunk_text(text: str, max_tokens: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks based on tokens."""
    tokens = encoding.encode(text)
    chunks, start = [], 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks

def embed_texts(chunks: list[str], model: str = "text-embedding-3-large") -> np.ndarray:
    """Create embeddings for text chunks."""
    vectors = []
    for i in range(0, len(chunks), 50):
        batch = chunks[i:i+50]
        res = client.embeddings.create(model=model, input=batch)
        vectors.extend([d.embedding for d in res.data])
    return np.array(vectors, dtype="float32")

def embed_query(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    res = client.embeddings.create(model=model, input=text)
    return np.array(res.data[0].embedding, dtype="float32")

# --- MAIN ENDPOINT ---

@app.post("/chat", response_model=QueryOut)
async def chat(query: QueryIn):
    """
    Accepts query + pdfurl from Bubble, 
    extracts info from zoning plan PDF, 
    and returns GPT answer.
    """

    start_time = time.time()

    # Step 1: Download PDF
    text = extract_text_from_pdf_url(query.pdfurl)
    if not text:
        raise HTTPException(status_code=400, detail="No text could be extracted from PDF.")

    # Step 3: Chunk text
    chunks = chunk_text(text)

    # Step 4: Embed chunks and create FAISS index
    embeddings = embed_texts(chunks)
    faiss.normalize_L2(embeddings)
    dim = len(embeddings[0])
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(embeddings)

    # Step 5: Embed query
    query_text = query.query
    if query.designation:
        query_text += f"\nZoning designations: {', '.join(query.designation)}"
    query_vec = embed_query(query_text).reshape(1, -1)
    faiss.normalize_L2(query_vec)

    # Step 6: Search top chunks
    distances, ids = index.search(query_vec, 10)
    top_chunks = [chunks[i] for i in ids[0]]
    context = "\n\n".join(top_chunks)

    # Step 7: Generate GPT response
    prompt = f"""
You are an expert zoning plan assistant.
Use the following zoning plan information and designations to answer clearly and concisely.

Parcel ID: {query.parcelid}
Plan ID: {query.planid}
Zoning designations: {', '.join(query.designation or [])}

CONTEXT:
{context}

QUESTION:
{query.query}

Answer in clear, direct language suitable for a property owner.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800,
    )

    answer = response.choices[0].message.content.strip()
    print(f"Processed {query.planid} in {time.time() - start_time:.2f}s")

    return {"answer": answer, "planid": query.planid}

