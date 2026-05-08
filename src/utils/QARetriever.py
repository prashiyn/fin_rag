import json
import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import uuid
try:
    from src.services.doc_processing_llm import DocProcessingEmbeddings
except ImportError:
    from services.doc_processing_llm import DocProcessingEmbeddings

class QAChromaLoader:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        persist_directory: Optional[str] = None,
        collection_name: str = "lotus_qa",
        embeddings_model_name: Optional[str] = None,
        chroma_server_host: Optional[str] = None,
        chroma_server_port: Optional[int] = None,
    ):
        self.collection_name = collection_name
        self.embeddings_model_name = embeddings_model_name or os.environ.get("EMBEDDINGS_MODEL_NAME", "BAAI/bge-m3")
        self.persist_directory = persist_directory or "./chroma_db"
        cfg = config or {
            "llm_service_base_url": os.environ.get("LLM_SERVICE_BASE_URL"),
            "llm_service_provider": os.environ.get("LLM_SERVICE_PROVIDER", "openai"),
            "embeddings_model_name": self.embeddings_model_name,
        }
        self.embedding_function = DocProcessingEmbeddings.from_config(cfg)

        settings = Settings(anonymized_telemetry=False, allow_reset=True)
        if chroma_server_host:
            port = int(chroma_server_port or 8000)
            self.client = chromadb.HttpClient(
                host=chroma_server_host,
                port=port,
                settings=settings,
            )
        else:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=settings,
            )

        self.collection = self._create_or_get_collection()
        
    def _create_or_get_collection(self):
        # This is a more robust, atomic operation.
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine",
                      "embedding_model": self.embeddings_model_name} 

        )
        print(f"Successfully got or created collection: {self.collection_name}")
        return collection
    
    def load_qa_data(self, qa_data: List[Dict[str, Any]], batch_size: int = 100):
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        print(f"Processing {len(qa_data)} QA entries...")
        
        for idx, qa_item in enumerate(qa_data):
            # Create document content and store the data as JSON string
            doc_content = json.dumps({
                "question": qa_item["question"],
                "question_rewritten": qa_item["question_rewritten"],
                "data": qa_item["data"]
            }, ensure_ascii=False)

            
            # Metadata
            metadata = {
                "doc_id": f"qa_{idx}",
                "prev_chunk_id": f"qa_{idx-1}" if idx > 0 else "", # leave blank if first or last item, no prev or next
                "next_chunk_id": f"qa_{idx+1}" if idx < len(qa_data) - 1 else "",
                "question": qa_item["question"],
                "question_rewritten": qa_item["question_rewritten"]
            }
            
            entry_id = f"qa_{uuid.uuid4().hex[:8]}_{idx}"
            
            documents.append(doc_content)
            metadatas.append(metadata)
            ids.append(entry_id)
            
            # Process in batches
            if len(documents) >= batch_size:
                self._add_batch(documents, metadatas, ids)
                documents, metadatas, ids = [], [], []
        
        if documents:
            self._add_batch(documents, metadatas, ids)
        
        print(f"Successfully loaded {len(qa_data)} QA entries into Chroma")
        
    def _add_batch(self, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Add a batch of documents to the collection."""
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Added batch of {len(documents)} documents")
        except Exception as e:
            print(f"Error adding batch: {e}")
            raise
    
    def query_qa(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        parsed_results = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                try:
                    qa_data = json.loads(doc)
                    qa_data['metadata'] = metadata
                    parsed_results.append(qa_data)
                except json.JSONDecodeError:
                    print(f"Error parsing document: {doc}")
                    
        return parsed_results
    
    def reset_collection(self): 
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted collection: {self.collection_name}")
            self.collection = self._create_or_get_collection()
        except Exception as e:
            print(f"Error resetting collection: {e}")



def load_qa_chroma_instance(
    qa_data_path: str,
    config: Optional[Dict[str, Any]] = None,
    persist_directory: Optional[str] = "./chroma_db",
    chroma_server_host: Optional[str] = None,
    chroma_server_port: Optional[int] = None,
    embeddings_model_name: Optional[str] = None,
):
    with open(qa_data_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    qa_loader = QAChromaLoader(
        config=config,
        persist_directory=persist_directory,
        collection_name="lotus_qa",
        embeddings_model_name=embeddings_model_name,
        chroma_server_host=chroma_server_host,
        chroma_server_port=chroma_server_port,
    )
    # qa_loader.reset_collection()
    qa_loader.load_qa_data(qa_data)    
    return qa_loader # in case inmediate usage is required -> qa_loader.query_qa("What is the sales volume in China", n_results=5)


if __name__ == "__main__":
    qa_data ="/root/autodl-tmp/dir_tzh/lotus_dataset/write_csv_json/input.json"
    qa_table_persist_directory = '/root/autodl-tmp/hyc_production/RAG_Agent/log/qa_chroma_bge/'

    # qa_loader = load_qa_chroma_instance(qa_data_path = qa_data, persist_directory = qa_table_persist_directory)
    qa_loader = QAChromaLoader(persist_directory=qa_table_persist_directory, collection_name="lotus_qa")

    # results = qa_loader.query_qa("What is the sales volume", n_results=3)
    # results = qa_loader.query_qa("What is Lotus Tech's service sales revenue?", n_results=3)
    # results = qa_loader.query_qa("What is the sports car sales volume for Lotus Technology in the second quarter of 2024?", n_results=3)
    # results = qa_loader.query_qa("What was the number of Lotus Technology stores in Q1 2023?", n_results=3)
    # results = qa_loader.query_qa("What was the number of Lotus Technology stores in the first quarter of 2023?", n_results=3)
    sample = qa_loader.collection.get(limit=1, include=['embeddings'])
    if sample['embeddings'].any():
        embedding_dim = len(sample['embeddings'][0])
        print(f"Embedding dimension: {embedding_dim}")
    else:
        print("No embeddings found in the collection.")
    
    query_question = "What was the number of Lotus Technology stores in the first quarter of 2023?"
    query_question = "What was Lotus Technology's sales revenue in the third quarter of 2023?"
    results = qa_loader.query_qa(query_question, n_results=3)
    print(f"Searching for: {query_question}")
    print("-"*50)

    qa_pairs = []
    for result in results:
        print(f"Question: {result['question']}")
        print(f"Rewritten: {result['question_rewritten']}")
        print(f"Data: {result['data']}")
        print(f"Metadata: {result['metadata']}")

        qa_pairs.append(
                        {
                            "question": result['question_rewritten'],
                            "answer": result['data']
                        }
                    )
        print("-" * 50)

    print(qa_pairs)    
        
    # qa_loader = QAChromaLoader(persist_directory="./chroma_db", collection_name="QA_chroma")
    # qa_loader.load_qa_data(your_qa_data_list)
