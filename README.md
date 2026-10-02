# 🤖 DocuMind AI — Lightweight RAG Chatbot

DocuMind AI is a lightweight Retrieval-Augmented Generation (RAG) chatbot that allows users to upload documents and ask questions about their content.

The application extracts text from uploaded documents, splits the content into meaningful chunks, retrieves the most relevant sections for a user's question, and uses the Groq API to generate an answer based on the retrieved context.

The project is designed to be lightweight and easy to deploy on free hosting platforms such as Render.

---

## 🚀 Live Demo

🌐 **Live Application:**  
https://ai-rag-2.onrender.com

---

## ✨ Features

- 📄 Upload PDF, DOCX, TXT and Markdown files
- 🔍 Lightweight document retrieval
- 🤖 AI-powered answers using Groq
- 💬 Ask questions about uploaded documents
- ⚡ Fast and lightweight backend
- 🌐 Flask-based web application
- ☁️ Free deployment using Render
- 🔐 API keys stored using environment variables
- 📦 No local LLM required
- 🧠 Retrieval-Augmented Generation architecture

---

## 🧠 How It Works

The application follows a simple RAG pipeline:

```text
                User
                 │
                 ▼
        Upload Document
                 │
                 ▼
        Text Extraction
                 │
                 ▼
         Text Chunking
                 │
                 ▼
      Lightweight Retrieval
                 │
                 ▼
      Relevant Context
                 │
                 ▼
          Groq LLM API
                 │
                 ▼
          AI Generated
             Answer
