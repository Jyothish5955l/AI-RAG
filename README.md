# DocuMind AI — Lightweight RAG

Flask + Groq + PDF/DOCX/TXT retrieval. This Render-friendly version removes Ollama, ChromaDB, LangChain, and local embedding models.

## Why this version deploys more easily
- No `venv/` or `.venv/` committed.
- No Ollama model download.
- No ChromaDB/Rust/ONNX runtime.
- Retrieval is lightweight TF-IDF-style lexical scoring implemented in pure Python.
- Groq handles generation through an API.
- Flask serves the frontend and backend from one Render Web Service.

## Local
```bash
py -3.13 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env`:
```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

Run:
```bash
python server.py
```
Open http://localhost:5000

## GitHub
```bash
git init
git add .
git commit -m "Deploy lightweight DocuMind RAG"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## Render
Create a Web Service from the GitHub repo:
- Build: `pip install -r requirements.txt`
- Start: `gunicorn server:app`
- Plan: Free
- Environment variable: `GROQ_API_KEY` = your Groq key

Do NOT upload `.env`, `venv/`, or `.venv/`.
