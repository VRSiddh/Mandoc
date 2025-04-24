# Mandoc: Multi-Format Document Question Answering Assistant

**Mandoc** is a powerful, extensible AI assistant that enables you to upload and chat with your own documents—across a wide range of formats. Built with Streamlit and advanced retrieval-augmented generation (RAG), Mandoc supports PDFs, DOCX, PPTX, XLSX, CSV, JSON, TXT, and image files (PNG, JPG), including OCR for scanned documents and images.

---

## 🚀 Features

- **Multi-format Upload:** Supports PDF, DOCX, PPTX, XLSX, CSV, JSON, TXT, PNG, JPG, and more.
- **AI-Powered Q&A:** Ask natural language questions about your documents and get accurate, context-aware answers.
- **Semantic Search:** Uses state-of-the-art embeddings and vector search for deep document understanding.
- **OCR Integration:** Extracts text from images and scanned documents using OCR.
- **Web Link Crawling:** Follows and indexes hyperlinks found within your documents.
- **Source Attribution:** Every answer cites the source file and page (where applicable).
- **Fast & Parallel Processing:** Efficiently handles large documents (even thousands of pages) with parallel and GPU-accelerated processing.
- **User-Friendly Interface:** Clean Streamlit UI for easy uploading, chatting, and management.

---

## 🛠️ How It Works

1. **Upload** your documents (any supported format).
2. **Processing:** The app extracts, chunks, and indexes content using semantic embeddings and FAISS vector search.
3. **Chat:** Ask questions in plain English; the app retrieves relevant chunks and generates answers using a large language model.
4. **Citations:** Answers always include source references for transparency.

---

## 📦 Installation & Usage

1. **Clone the repo and install requirements:**
pip install -r requirements.txt

2. **Run the app:**
streamlit run app.py

3. **Open your browser and start uploading documents!**

---

## 📚 Use Cases

- Academic research: Instantly query large collections of papers, books, or notes.
- Business intelligence: Search and summarize contracts, reports, or meeting notes.
- Legal, compliance, and more.

---

**Mandoc** empowers you to unlock insights from your own files—securely, privately, and with full source traceability.

---

*Inspired by best practices in document QA and RAG-powered assistants.*
