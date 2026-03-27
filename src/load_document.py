import hashlib
import os
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from services.bm25_storage import get_bm25_local_dir, upload_bm25_to_s3
from utils.bm25Retriever import load_from_chroma_and_save


class Chunk(BaseModel):
    """Canonical chunk object: content (original text/table/image), type, and metadata."""

    chunk_id: str
    content: str
    type: Literal["text", "table", "image"]
    doc_id: str
    page: int | None = None
    bundle_id: str
    section_title: str | None = None
    title_summary: str = ""
    publish_date: str | None = None
    prev_chunk: str | None = None
    next_chunk: str | None = None


class LoadDataContent(BaseModel):
    page_start: int = Field(..., description="Start page for ingest filter")
    page_end: int = Field(..., description="End page for ingest filter")
    page_date_published: str | None = Field(None, description="Document publish date")
    chunks: list[Chunk]


class LoadDataRequest(BaseModel):
    collection: str
    data: LoadDataContent


def _to_date_obj(raw: str | None) -> datetime:
    if not raw:
        return datetime.min
    value = raw.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return datetime.min


def _to_iso_date(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%Y/%m/%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_data_into_collection(rag_manager: Any, config: dict, payload: LoadDataRequest) -> dict:
    """
    Ingest chunk payload into a collection with the same flow as script/load_data.py:
    - ensure collection exists
    - reset main and title-summary Chroma stores
    - page-range filter + dedup by content hash (newer publish date wins)
    - generate prev_chunk_id / next_chunk_id
    - store chunks and title summaries into Chroma
    - rebuild BM25 index from Chroma documents
    - refresh retriever for this collection
    """
    collection_name = payload.collection.strip()
    if not collection_name:
        raise ValueError("collection is required")

    created_new = collection_name not in rag_manager._collections
    rag_manager.create_collection(collection_name)
    chroma, ts_chroma = rag_manager._collections[collection_name]
    chroma.reset_collection()
    ts_chroma.reset_collection()

    start_page = payload.data.page_start
    end_page = payload.data.page_end
    default_publish_date = _to_iso_date(payload.data.page_date_published)
    global_id = 0
    title_summaries: set[str] = set()
    content_dict: dict[str, tuple[str, dict[str, Any], datetime]] = {}

    for chunk in payload.data.chunks:
        page = chunk.page
        if page is not None and not (start_page <= page <= end_page):
            continue

        content = chunk.content or ""
        doc_hash = _content_hash(content)
        publish_date = _to_iso_date(chunk.publish_date) or default_publish_date
        publish_dt = _to_date_obj(publish_date)

        metadata: dict[str, Any] = {
            "filename": chunk.doc_id or f"{collection_name}.json",
            "page_number": page,
            "date_published": publish_date,
            "doc_id": doc_hash,
            "global_id": global_id,
            "bundle_id": chunk.bundle_id,
            "title_summary": chunk.title_summary or "",
            "section_title": chunk.section_title,
            "chunk_type": chunk.type,
            "source_chunk_id": chunk.chunk_id,
            "source_doc_id": chunk.doc_id,
            "source_prev_chunk": chunk.prev_chunk,
            "source_next_chunk": chunk.next_chunk,
        }
        global_id += 1
        if metadata["title_summary"]:
            title_summaries.add(metadata["title_summary"])

        if doc_hash in content_dict:
            _, existing_meta, existing_dt = content_dict[doc_hash]
            # Keep the newest publish date entry (same behavior as script/load_data.py)
            if publish_dt > existing_dt:
                content_dict[doc_hash] = (content, metadata, publish_dt)
        else:
            content_dict[doc_hash] = (content, metadata, publish_dt)

    ordered = list(content_dict.values())
    contents = [item[0] for item in ordered]
    metadatas = [item[1] for item in ordered]
    doc_ids = [m["doc_id"] for m in metadatas]

    for i in range(len(metadatas)):
        if i > 0 and metadatas[i]["filename"] == metadatas[i - 1]["filename"]:
            metadatas[i]["prev_chunk_id"] = doc_ids[i - 1]
        else:
            metadatas[i]["prev_chunk_id"] = ""
        if i < len(metadatas) - 1 and metadatas[i]["filename"] == metadatas[i + 1]["filename"]:
            metadatas[i]["next_chunk_id"] = doc_ids[i + 1]
        else:
            metadatas[i]["next_chunk_id"] = ""

    batch_size = int(config.get("load_data_batch_size", 100))
    title_list = list(title_summaries)
    for i in range(0, len(title_list), batch_size):
        ts_chroma.add_texts(texts=title_list[i : i + batch_size])

    for i in range(0, len(contents), batch_size):
        chroma.add_texts(
            texts=contents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
            ids=doc_ids[i : i + batch_size],
        )

    documents = rag_manager.get_collection_documents(collection_name)
    bm25_save_dir = get_bm25_local_dir(config, collection_name)
    load_from_chroma_and_save(documents, bm25_save_dir)
    upload_bm25_to_s3(config, collection_name, bm25_save_dir)

    top_k = int(config.get("load_data_default_top_k", 10))
    rag_manager.upsert_collection_retriever(collection_name, top_k)

    return {
        "collection": collection_name,
        "created_collection": created_new,
        "stored_chunks": len(contents),
        "title_summaries": len(title_list),
        "top_k": top_k,
    }
