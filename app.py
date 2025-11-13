import os
import io
import json
import time
import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from starlette.middleware.cors import CORSMiddleware

# --- CONFIG ---
OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)

# Update this to your GitHub repo raw URL
GITHUB_BASE = "https://raw.githubusercontent.com/gpurva/Landing_Chatbot/main/faiss_indexes"

MANIFEST_URL = f"{GITHUB_BASE}/faiss_manifest.json"

# --- Load manifest ---
try:
    manifest_resp = requests.get(MANIFEST_URL, timeout=20)
    manifest_resp.raise_for_status()
    MANIFEST = manifest_resp.json()
    UZIP_MAP = MANIFEST["uzip_mapping"]
    GLOBAL_PDFS = MANIFEST.get("global_pdfs", [])
except Exception as e:
    raise RuntimeError(f"❌ Failed to load manifest from GitHub: {e}")

# --- Cache ---
INDEX_CACHE = {}
CHUNKS_CACHE = {}

# --- FASTAPI APP ---
app = FastAPI(title="Zoning Plan Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class QueryIn(BaseModel):
    uzip_id: str
    query: str

class QueryOut(BaseModel):
    answer: str


# --- UTIL FUNCTIONS ---
def download_file_from_github(filename: str) -> bytes:
    """Download a file from GitHub raw link."""
    url = f"{GITHUB_BASE}/{filename}"
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        raise HTTPException(status_code=404, detail=f"File not found on GitHub: {filename}")
    return resp.content


def read_index_from_bytes(index_bytes: bytes):
    """Load FAISS index directly from bytes."""
    index_stream = io.BytesIO(index_bytes)
    return faiss.read_index(index_stream)


def load_faiss_and_chunks(base_name: str):
    """Loads FAISS index + chunks (cached). Handles both '.pdf' and non-pdf names."""
    if base_name in INDEX_CACHE and base_name in CHUNKS_CACHE:
        return INDEX_CACHE[base_name], CHUNKS_CACHE[base_name]

    index_filename = f"{base_name}.index"
    chunks_filename = f"{base_name}_chunks"  # no .json

    # Download
    index_bytes = download_file_from_github(index_filename)
    chunks_bytes = download_file_from_github(chunks_filename)

    # Parse
    index = read_index_from_bytes(index_bytes)
    chunks = json.loads(chunks_bytes.decode("utf-8"))

    INDEX_CACHE[base_name] = index
    CHUNKS_CACHE[base_name] = chunks
    return index, chunks


def embed_query(text: str, model: str = "text-embedding-3-large") -> np.ndarray:
    res = client.embeddings.create(model=model, input=text)
    vec = np.array(res.data[0].embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def search_index(index, chunks, query_vec, top_k=5):
    distances, ids = index.search(query_vec, top_k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]


# --- MAIN ENDPOINT ---
@app.post("/chat", response_model=QueryOut)
async def chat(query: QueryIn):
    """
    Accepts uzip_id + query.
    Loads FAISS + chunks from GitHub for that uzip_id’s PDF (with .pdf suffix),
    adds building_law + technical_conditions (no .pdf suffix),
    searches, and returns GPT-generated answer.
    """
    start_time = time.time()
    uzip_id = query.uzip_id.strip()

    if uzip_id not in UZIP_MAP:
        raise HTTPException(status_code=404, detail=f"uzip_id '{uzip_id}' not found in manifest")

    pdf_name = UZIP_MAP[uzip_id]  # e.g., "brwinow_A.pdf"

    try:
        index_main, chunks_main = load_faiss_and_chunks(pdf_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load FAISS for {pdf_name}: {e}")

    # Embed query
    query_vec = embed_query(query.query)
    top_main = search_index(index_main, chunks_main, query_vec, top_k=7)

    # Add global PDFs
    global_context = []
    for gpdf in GLOBAL_PDFS:
        try:
            gindex, gchunks = load_faiss_and_chunks(gpdf)
            gtop = search_index(gindex, gchunks, query_vec, top_k=3)
            global_context.extend(gtop)
        except Exception as e:
            print(f"⚠️ Skipping global file {gpdf}: {e}")

    # Combine context
    context = "\n\n".join(top_main + global_context)

    # Compose prompt
    prompt = f"""
        You are a zoning and building regulation assistant.
        Use the following context extracted from zoning plan documents and general building laws to answer the question accurately and clearly.

        UZIP ID: {uzip_id}
        Document: {pdf_name}

        Context:
        {context}

        Question:
        {query.query}

        Answer concisely and in simple language understandable by a property owner.
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=700,
    )

    answer = response.choices[0].message.content.strip()
    print(f"✅ {uzip_id} answered in {time.time() - start_time:.2f}s")

    return {"answer": answer}
