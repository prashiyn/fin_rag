import os
import logging
logger = logging.getLogger(__name__)
import torch
import chromadb
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

from services.bm25_storage import download_bm25_from_s3_if_needed, get_bm25_local_dir
try:
    from src.services.doc_processing_llm import DocProcessingEmbeddings
except ImportError:
    from services.doc_processing_llm import DocProcessingEmbeddings
from .ensembleRetriever import EnsembleRetriever
import GPUtil

class RAGManager:
    """Singleton class for managing RAG collections"""
    _collections: Dict[str, Tuple[Chroma, Chroma]] = {}
    _retrievers: List[EnsembleRetriever] = []
    _retriever_by_collection: Dict[str, EnsembleRetriever] = {}

    _instance = None
    _config = None

    def __new__(cls, config: Dict = None, collections: Dict[str, int] = None):
        if cls._instance is None:
            if config is None:
                logger.error("No config provided")
                raise ValueError("No config provided for RAGManager")
            cls._instance = super(RAGManager, cls).__new__(cls)
            cls._instance._initialize(config, collections)
        return cls._instance

    def __init__(self, config: Dict = None, collections: Dict[str, int] = None):
        pass

    def _initialize(self, config: Dict, collections: Dict[str, int]):
        self._config = config
        self.embeddings_model_name = config['embeddings_model_name']
        self.batch_size = 5
        self._collections = {}
        self._retrievers = []
        self._retriever_by_collection = {}

        try:
            logger.info("Initializing LLM service embedding client...")
            self.embeddings = DocProcessingEmbeddings.from_config(config)
            logger.info("Embedding client initialized successfully.")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding client: {e}")

        if collections is not None:
            for collection, top_k in collections.items():
                if top_k <= 0:
                    continue
                self.create_collection(collection)
                self.upsert_collection_retriever(collection, int(top_k))

    def _chroma_kwargs(self, chroma_collection_name: str, subpath: str):
        """Return kwargs for LangChain Chroma: client-server (host/port) or local (persist_directory)."""
        host = self._config.get("chroma_server_host")
        port = self._config.get("chroma_server_port", 8000)
        if host:
            return {
                "collection_name": chroma_collection_name,
                "embedding_function": self.embeddings,
                "host": host,
                "port": int(port),
                "relevance_score_fn": "cosine",
            }
        persist_dir = os.path.join(self._config["persist_directory"], subpath)
        return {
            "collection_name": chroma_collection_name,
            "embedding_function": self.embeddings,
            "persist_directory": persist_dir,
            "relevance_score_fn": "cosine",
        }

    def create_collection(self, collection_name: str):
        """Create a new collection with all supported retrievers"""
        if collection_name not in self._collections:
            # Main chunks: collection_name. Title-summary: collection_name_ts (same name in local mode per subpath).
            chroma = Chroma(**self._chroma_kwargs(collection_name, "chroma"))
            ts_collection = f"{collection_name}_ts" if self._config.get("chroma_server_host") else collection_name
            ts_chroma = Chroma(**self._chroma_kwargs(ts_collection, "ts_chroma"))
            self._collections[collection_name] = (chroma, ts_chroma)
            logger.warning("Load Chroma: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))

    def get_collection_documents(self, collection_name: str, doc_ids: Optional[List[str]] = None) -> List[Document]:
        """Get documents from a collection by document IDs. User should not assume that the order of the returned documents matches the order of the input IDs."""
        chroma, _ = self._collections[collection_name]
        if doc_ids is None:
            chroma_docs = chroma.get()
        else:
            chroma_docs = chroma.get(ids=doc_ids)

        documents = [
            Document(
                page_content=page_content,
                metadata=metadata
            )
            for page_content, metadata in zip(chroma_docs['documents'], chroma_docs['metadatas'])
        ]
        return documents
    
    def create_retriever(self, k: int, collection_name: str, retriever_type: str = "chroma"):
        """Create a specific retriever for a collection"""
        if collection_name not in self._collections:
            raise ValueError(f"Collection {collection_name} does not exist")
            
        bm25_dir = get_bm25_local_dir(self._config, collection_name)
        download_bm25_from_s3_if_needed(self._config, collection_name, bm25_dir)

        chroma, ts_chroma = self._collections[collection_name]
        retriver = EnsembleRetriever(bm25_dir, chroma, ts_chroma, k, self.embeddings, enable_expand = True)
            
        return retriver

    def has_collection(self, collection_name: str) -> bool:
        return collection_name in self._collections

    def get_retriever(self, collection_name: str) -> EnsembleRetriever:
        retriever = self._retriever_by_collection.get(collection_name)
        if retriever is None:
            raise ValueError(f"Collection {collection_name} is not initialized")
        return retriever

    def upsert_collection_retriever(self, collection_name: str, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if collection_name not in self._collections:
            self.create_collection(collection_name)
        retriever = self.create_retriever(top_k, collection_name, retriever_type="ensemble")
        self._retriever_by_collection[collection_name] = retriever
        self._retrievers = list(self._retriever_by_collection.values())

    def hydrate_from_chroma(self, default_top_k: int = 10, topk_map: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        if default_top_k <= 0:
            raise ValueError("default_top_k must be > 0")
        discovered = self.discover_collections()
        resolved: Dict[str, int] = {}
        for name in discovered:
            top_k = int((topk_map or {}).get(name, default_top_k))
            if top_k <= 0:
                continue
            self.create_collection(name)
            self.upsert_collection_retriever(name, top_k)
            resolved[name] = top_k
        return resolved

    def discover_collections(self) -> List[str]:
        """
        Discover existing main collections from Chroma.
        - Server mode: list all collections and skip *_ts title-summary collections.
        - Local mode: list collections from persist_directory/chroma.
        """
        host = self._config.get("chroma_server_host")
        port = int(self._config.get("chroma_server_port", 8000))
        settings = chromadb.config.Settings(anonymized_telemetry=False, allow_reset=True)
        if host:
            client = chromadb.HttpClient(host=host, port=port, settings=settings)
            names = [c.name for c in client.list_collections()]
            return sorted([n for n in names if not n.endswith("_ts")])

        persist_dir = os.path.join(self._config["persist_directory"], "chroma")
        client = chromadb.PersistentClient(path=persist_dir, settings=settings)
        names = [c.name for c in client.list_collections()]
        return sorted(names)


# Usage example
def main():
    from config import get_config
    config = get_config()
    
    questions = [
        "Are there any new releases in 2023?",
        "Can you tell me how Lotus's approach to vehicle design evolved between 2000 and 2020?",
        "What are the unique technical features that make Lotus stand out in racing?" ,
        "Can you explain the lightweight design philosophy of Lotus?" ,
        "Which Lotus models are best known for their driving performance on the track?" ,
    ]
    
    rag = RAGManager(config)
    log_gpu_usage('RAGManager init')
    rag.create_collection("lotus")
    #rag.create_collection("zeekr")
    log_gpu_usage('RAGManager create collection')
    retriever = rag.create_retriever(5, "lotus", "ensemble")
    #retriever = rag.create_retriever(5, "zeekr", "ensemble")
    log_gpu_usage('RAGManager get retriever')

    for q in questions:
        documents = retriever.invoke(q)
        log_gpu_usage('RAGManager invoke retriever')
        print(f"Question: {q}")
        for i, doc in enumerate(documents):
            print(f"{i}: {doc}")
        print("")
        

def log_gpu_usage(event_name):
    gpus = GPUtil.getGPUs()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    gpu_log_file = "gpu_usage.log"
    for gpu in gpus:
        gpu_info = (
            f"Timestamp: {timestamp}, Event: {event_name}, "
            f"GPU ID: {gpu.id}, GPU Name: {gpu.name}, "
            f"Memory Used: {gpu.memoryUsed} MB, Memory Total: {gpu.memoryTotal} MB"
        )
        # 将信息追加到日志文件
        with open(gpu_log_file, 'a') as f:
            f.write(gpu_info + '\n')

if __name__ == "__main__":
    main()
