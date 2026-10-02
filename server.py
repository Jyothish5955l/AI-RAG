# DocuMind AI - lightweight RAG backend for Render
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from PyPDF2 import PdfReader
from datetime import datetime
import os
import re
import math
import tempfile

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env locally or Render Environment Variables.")

app = Flask(__name__, static_folder=".")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

client = Groq(api_key=GROQ_API_KEY)

chunks_store = []
idf = {}
current_document = {
    "filename": None,
    "pages": 0,
    "chunks": 0,
    "characters": 0,
    "file_size": 0,
    "uploaded_at": None
}

STOP_WORDS = {
    "the","a","an","and","or","but","if","then","than","of","to","in","on","for",
    "with","from","by","is","are","was","were","be","been","being","as","at","it",
    "this","that","these","those","i","you","he","she","we","they","what","which",
    "who","when","where","why","how","do","does","did","can","could","should",
    "would","will","about","into","over","under","after","before","not"
}

def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9_]{2,}", text.lower())
    return [w for w in words if w not in STOP_WORDS]

def build_index(chunks):
    global idf
    document_frequency = {}
    for item in chunks:
        terms = set(tokenize(item["text"]))
        for term in terms:
            document_frequency[term] = document_frequency.get(term, 0) + 1
    n = max(len(chunks), 1)
    idf = {
        term: math.log((1 + n) / (1 + df)) + 1
        for term, df in document_frequency.items()
    }

def score_chunk(text, question):
    q_terms = tokenize(question)
    if not q_terms:
        return 0.0
    terms = tokenize(text)
    if not terms:
        return 0.0

    counts = {}
    for term in terms:
        counts[term] = counts.get(term, 0) + 1

    score = 0.0
    for term in set(q_terms):
        if term in counts:
            tf = 1 + math.log(counts[term])
            score += tf * idf.get(term, 1.0)
    # Small phrase-match bonus.
    q = " ".join(q_terms)
    normalized = " ".join(terms)
    if len(q) > 8 and q in normalized:
        score += 2.0
    return score

def retrieve(question, k=5):
    scored = []
    for item in chunks_store:
        scored.append((score_chunk(item["text"], question), item))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [item for score, item in scored if score > 0][:k]
    if not selected:
        selected = [item for _, item in scored[:k]]
    return selected

def extract_document(file_path, filename):
    documents = []
    page_count = 0
    total_characters = 0
    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        reader = PdfReader(file_path)
        page_count = len(reader.pages)
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                total_characters += len(text)
                documents.append({"text": text, "page": page_number, "source": filename})

    elif lower_name.endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
        if text:
            total_characters = len(text)
            documents.append({"text": text, "page": 1, "source": filename})
        page_count = 1

    elif lower_name.endswith(".docx"):
        from docx import Document
        doc = Document(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
        if text:
            total_characters = len(text)
            documents.append({"text": text, "page": 1, "source": filename})
        page_count = 1

    else:
        raise ValueError("Unsupported file type. Use PDF, TXT, DOCX or MD.")

    return documents, page_count, total_characters

def split_documents(documents, chunk_size=1200, overlap=200):
    result = []
    for doc in documents:
        text = doc["text"]
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                result.append({
                    "text": piece,
                    "page": doc["page"],
                    "source": doc["source"]
                })
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return result

@app.get("/")
def home():
    return send_from_directory(".", "index.html")

@app.get("/api")
def api_home():
    return jsonify({
        "status": "running",
        "message": "DocuMind AI RAG API is live!",
        "model": GROQ_MODEL
    })

@app.get("/status")
def status():
    return jsonify({
        "documents_loaded": bool(chunks_store),
        "models_ready": True,
        "document": current_document
    })

@app.post("/upload")
def upload():
    global chunks_store, current_document
    if "file" not in request.files:
        return jsonify({"error": "No file found."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    tmp_path = None
    try:
        filename = os.path.basename(file.filename)
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        file_size = os.path.getsize(tmp_path)
        documents, page_count, total_characters = extract_document(tmp_path, filename)
        if not documents:
            return jsonify({"error": "Could not extract any text from this document."}), 400

        chunks_store = split_documents(documents)
        build_index(chunks_store)

        current_document = {
            "filename": filename,
            "pages": page_count,
            "chunks": len(chunks_store),
            "characters": total_characters,
            "file_size": file_size,
            "uploaded_at": datetime.now().strftime("%d %b %Y, %I:%M:%S %p")
        }

        return jsonify({
            "success": True,
            **current_document,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "message": f"'{filename}' uploaded and indexed successfully."
        })

    except Exception as e:
        print("Upload error:", e)
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

@app.post("/chat")
def chat():
    if not chunks_store:
        return jsonify({"error": "No document loaded. Upload a document first."}), 400

    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        lower_question = question.lower()
        overview_words = [
            "what is inside", "what's inside", "what does this document contain",
            "what is this document about", "what does the document discuss",
            "summarize", "summary", "overview", "main topics",
            "tell me about the document"
        ]
        is_overview = any(word in lower_question for word in overview_words)

        docs = retrieve(question, k=12 if is_overview else 5)

        selected = []
        total_context_chars = 0
        max_context_chars = 30000
        for doc in docs:
            if total_context_chars + len(doc["text"]) > max_context_chars:
                break
            selected.append(doc)
            total_context_chars += len(doc["text"])

        context_parts = []
        for i, doc in enumerate(selected, 1):
            context_parts.append(
                f"SOURCE {i}\nPAGE: {doc['page']}\n\n{doc['text']}"
            )
        context = "\n\n--------------------\n\n".join(context_parts)

        if is_overview:
            instruction = """Give a useful overview of the document. Mention what it is about,
the main topics/sections, important points, and overall purpose/conclusion if available.
Use only the supplied document context. Do not invent information."""
        else:
            instruction = """Answer the user's question using ONLY the document context.
If the answer is not present, say: "I don't have enough information in the document to answer this."
Do not use outside knowledge or invent facts."""

        prompt = f"""You are DocuMind AI, a document question-answering assistant.

{instruction}

DOCUMENT:
{current_document["filename"]}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        answer = response.choices[0].message.content

        sources = []
        for i, doc in enumerate(selected, 1):
            sources.append({
                "id": i,
                "page": doc["page"],
                "content": doc["text"],
                "preview": doc["text"][:180] + ("..." if len(doc["text"]) > 180 else "")
            })

        return jsonify({
            "answer": answer,
            "sources": sources,
            "question": question,
            "document": current_document["filename"]
        })

    except Exception as e:
        print("Chat error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
