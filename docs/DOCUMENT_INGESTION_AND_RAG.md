# Document Ingestion and RAG Storage — Codebase Overview

This document describes how documents are ingested, chunked, stored in Chroma (and BM25), and used for RAG in this project. It focuses on **src/** and the boundary with **script/** so you can add APIs for collections and documents.

---

## 1. High-Level Flow

```
[External pipeline: PDF/DOCX → chunks + metadata]
         ↓
    JSON files (per-document or per-batch)
         ↓
script/load_data.py  (only place that writes into Chroma/BM25)
         ↓
    Chroma (main + title-summary) + BM25 index on disk
         ↓
At runtime: src/ only READS (RAGManager, EnsembleRetriever, ChatService)
         ↓
    Q&A API (no API to add collections or add documents)
```

- **Ingestion and chunking:** Done **outside** this repo. This repo **consumes pre-chunked JSON**.
- **Writing to Chroma/BM25:** Done only in **script/load_data.py** (and optionally script/QARetriever for a separate QA Chroma collection).
- **Reading for RAG:** Done in **src/** via RAGManager → EnsembleRetriever → Chroma + FAISS + BM25.

---

## 2. Where Documents Are Ingested (Not in src/)

Ingestion into Chroma and BM25 happens in **script/load_data.py**, not in the FastAPI app.

### 2.1 Entrypoint (script/load_data.py)

- Loads configuration from `.env` via `src/config.py`.
- Instantiates **RAGManager(config)** (from src).
- For each **collection directory** in a hardcoded list:
  - **create_collection(collection_name)** on RAGManager (creates empty Chroma collections).
  - **import_collection_from_dir(rag_manager, collection_name, dir_path, batch_size)** — does the actual load (see below).
  - **get_collection_documents(collection_name)** to fetch all docs from Chroma.
  - **load_from_chroma_and_save(documents, bm25_save_dir)** to build and save the BM25 index under `persist_directory/bm25_index/<collection_name>/`.

So: **collections and document sets are fixed at script run time**; the server never creates a collection or adds documents.

### 2.2 Document Format (Input to load_data)

- **Source:** A **directory of JSON files**. Only `*.json` is processed.
- **Structure of each file:** A single **JSON array**. Loaded with:
  - `JSONLoader(file_path=json_file, jq_schema=".[]", text_content=False)` (LangChain).
- **Semantics:**
  - **First element** of the array = **page-range metadata** (one object):
    - `start`, `end`: page range (numeric).
    - `date_published`: string (used for dedup and ordering).
  - **Remaining elements** = **one chunk per element**. Each is a JSON object:
    - **content** (string): text of the chunk.
    - **page_number** (number).
    - **bundle_id** (optional): groups chunks (e.g. same table/section). Used by the retriever to return whole bundles.
    - **title_summary** (optional): short summary for the “title-summary” retrieval path.

So **chunking is not done in this repo**; chunks are already in the JSON. An external pipeline (e.g. Unstructured + your own logic) would produce these JSON files.

### 2.3 What import_collection_from_dir Does

- Resets the two Chroma collections for that `collection_name` (main + title-summary).
- For each JSON file:
  - Parses page range from element 0.
  - For elements 1..N: extracts `content`, `page_number`, `bundle_id`, `title_summary`.
  - Optionally filters by page range (`page_number` in `[start, end]`).
  - Deduplicates by **SHA-256(content)**; on duplicate, keeps the chunk with **newer date_published**.
  - Builds metadata: `filename`, `page_number`, `date_published`, `doc_id` (= content hash), `global_id`, and later `prev_chunk_id` / `next_chunk_id` for same-file adjacency.
- **Title-summary store:** Collects unique `title_summary` strings and does **ts_chroma.add_texts(texts=title_summaries)** (in batches).
- **Main store:** **chroma.add_texts(texts=batch_contents, metadatas=batch_metadata, ids=batch_doc_ids)** in batches (ids = content hash).
- After that, BM25 index is built from Chroma documents (see above).

So **all writes to Chroma and BM25** go through this script; **src/** has no `add_texts` or equivalent for the main RAG collections.

---

## 3. What Lives in src/ — Read-Only RAG

### 3.1 RAGManager (src/utils/ragManager.py)

- **Singleton.** Initialized once at server startup with **config** and **collections** (e.g. `{"lotus": 10}`).
- **create_collection(collection_name)**  
  - Creates two LangChain Chroma instances:
    - **Main:** `Chroma(collection_name=collection_name, ...)` (subpath `"chroma"`).
    - **Title-summary:** `Chroma(collection_name=collection_name_ts, ...)` (subpath `"ts_chroma"`).
  - Uses **config**: `chroma_server_host` / `chroma_server_port` (client-server) or `persist_directory` (local). Embeddings from **config** `embeddings_model_name`.
- **get_collection_documents(collection_name, doc_ids=None)**  
  - Reads from the **main** Chroma collection (all or by ids) and returns a list of LangChain **Document** (page_content + metadata).
- **create_retriever(k, collection_name, retriever_type="ensemble")**  
  - Builds an **EnsembleRetriever** for that collection (BM25 dir = `persist_directory/bm25_index/<collection_name>`).

So in **src/**:
- RAGManager **creates** empty Chroma collections (and retrievers) when the process starts.
- It does **not** add documents; it only **reads** via `get_collection_documents` and via the retrievers.

### 3.2 EnsembleRetriever (src/utils/ensembleRetriever.py)

- **Inputs:** BM25 index directory, main **Chroma**, title-summary **Chroma**, k, embeddings, and options (e.g. `enable_expand=True`).
- **At init:**  
  - Loads **all** documents/embeddings from main Chroma: `chroma.get(include=["metadatas", "embeddings"])`.  
  - Builds **FAISS** index in memory from those embeddings (FaissRetriever).  
  - Loads title-summary Chroma and builds a second FAISS (title_summary_faiss_retriever).  
  - Loads **BM25** index from disk (BM25Retriever).  
  - Builds `docid2idx`, `chunk_metadata`, `title_summaries` for lookups.
- **invoke(query, hyde_chunks):**  
  - Runs **FAISS** (main + HyDE chunks), **title-summary FAISS**, and **BM25**.  
  - For each hit, optionally expands with **prev_chunk_id** / **next_chunk_id** (same document).  
  - Groups by **bundle_id** when present (returns whole bundle).  
  - Returns a list of chunk dicts (`page_content`, `metadata`, `bundle_id`, `retriever`, `score`).

So retrieval is **read-only**: it assumes Chroma and BM25 are already populated (by the script).

### 3.3 BM25 (src/utils/bm25Retriever.py)

- **load_from_chroma_and_save(documents, save_dir):**  
  - Takes a list of LangChain Documents (e.g. from `get_collection_documents`).  
  - Uses **page_content** as corpus and **metadata["doc_id"]** as doc ids.  
  - Tokenizes with **bm25s** (stopwords + stemmer), builds BM25 index, **saves to disk** under `save_dir`.  
- **BM25Retriever:** Loads that index from disk and returns (ids, scores) for a query.  
- **Used by:** script/load_data (to build the index after Chroma is filled) and by EnsembleRetriever (to run BM25 at query time).

### 3.4 Chroma Usage in src/

- **RAGManager** uses **langchain_chroma.Chroma** with:
  - Either **host + port** (Chroma server) or **persist_directory** (local).
  - **Embedding function** = HuggingFaceEmbeddings(config `embeddings_model_name`).
- **No** `add_texts` / `add_documents` in src/ for the main or title-summary RAG collections. The only code that adds to them is **script/load_data.py**.

### 3.5 QA Chroma (Separate Path)

- **QARetriever.QAChromaLoader** (src/utils/QARetriever.py) uses **Chroma** for a **different** use case: “lotus_qa” style Q&A (question → rewritten question + data). It has **load_qa_data** and **add**-style logic, but it is **not** used for the main RAG document store; it’s used by ChatService for a separate QA lookup. The main RAG path is: **RAGManager → EnsembleRetriever → main Chroma + ts Chroma + BM25**.

---

## 4. Server Startup and Collections

- In **server.py** lifespan:
  - **collections = {"lotus": 10}** is hardcoded.
  - **RAGManager(config=config, collections=collections)** is called.
  - That triggers **create_collection("lotus")** and **create_retriever(10, "lotus", "ensemble")** for the singleton.
- So at runtime there is **exactly one** collection name (“lotus”) and one retriever. There is **no API** to:
  - Add a new collection.
  - Add or update documents in a collection.
  - Trigger re-indexing of BM25.

---

## 5. End-to-End Data Flow Summary

| Stage | Where | What |
|-------|--------|------|
| **Chunking** | Outside repo | External pipeline produces JSON with one object per chunk (content, page_number, bundle_id, title_summary). |
| **Load** | script/load_data.py | Reads JSON dir → dedup → chroma.add_texts (main + ts) → get_collection_documents → load_from_chroma_and_save (BM25). |
| **Storage** | Chroma (server or local) + disk | Main collection, title-summary collection, BM25 index under `persist_directory/bm25_index/<collection_name>`. |
| **Retrieval** | src/ | RAGManager (singleton) → EnsembleRetriever (FAISS + title-summary FAISS + BM25) → rerank → LLM. |
| **APIs** | src/server.py | Q&A only (/api_chat, /api_chat_stream). No “create collection” or “ingest document” endpoints. |

---

## 6. What You Need to Add for “Add Collection / Add Documents” APIs

To support adding collections and documents from the API you will need to:

1. **Chunking (if you accept raw docs):**  
   Either:
   - Add a chunking step in **src/** (e.g. using Unstructured or LangChain text splitters), and produce the same JSON shape (content, page_number, bundle_id, title_summary), or  
   - Accept **already chunked** payloads (e.g. JSON) and skip chunking.

2. **Collection lifecycle:**  
   - Expose **create_collection** (or equivalent) via an API that calls **RAGManager.create_collection(collection_name)** and **create_retriever(...)** and stores the mapping (e.g. collection_name → top_k) in config or DB.  
   - Optionally support **delete / reset** collection.

3. **Document ingestion in src/:**  
   - Implement an ingestion path that:
     - Takes chunks (or raw docs + chunking),
     - Builds metadata (doc_id, global_id, prev/next_chunk_id, bundle_id, title_summary, etc.),
     - Calls **chroma.add_texts(...)** and **ts_chroma.add_texts(...)** for the right collection (from RAGManager._collections).
   - After adding documents, either:
     - Rebuild BM25: **get_collection_documents** → **load_from_chroma_and_save**, and re-create **EnsembleRetriever** for that collection, or  
     - Implement incremental BM25 updates if the library supports it.

4. **RAGManager/EnsembleRetriever:**  
   - Today **EnsembleRetriever** is built once at init and loads **all** Chroma data into FAISS. If you add documents at runtime, you must either:
     - Recreate the retriever (and FAISS index) after each batch, or  
     - Refactor to support incremental FAISS/Chroma updates and avoid loading the full collection into memory at init.

5. **Config / discovery:**  
   - Replace or extend the hardcoded **collections = {"lotus": 10}** with something dynamic (e.g. from DB or config) so new collections are visible to the server and retrievers.

This document gives you the full picture of how ingestion, chunking, and storage work today and what to extend in **src/** and **script/** to add collection and document APIs.
