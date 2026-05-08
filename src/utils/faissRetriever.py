import logging
logger = logging.getLogger(__name__)
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
try:
    from src.services.doc_processing_llm import DocProcessingEmbeddings
except ImportError:
    from services.doc_processing_llm import DocProcessingEmbeddings

class FaissRetriever:
    """Faiss retriever compatible with LangChain that supports metadata filtering."""
    
    def __init__(self, embeddings, embedding_fn: Any):
        super().__init__()
        self.embeddings = embedding_fn
        embeddings = np.array(embeddings)
        dimension = embeddings.shape[1]

        # res = faiss.StandardGpuResources()
        self.index = faiss.IndexFlatIP(dimension)
        # self.index = faiss.index_cpu_to_gpu(res, 0, index)

        x = embeddings.astype('float32')
        faiss.normalize_L2(x)

        self.index.add(x)
        
        logger.info(f"Building FAISS index with {len(embeddings)} vectors of dimension {dimension}")

    def invoke(
            self,
            querys: list[str],
            k: int
        ):
        query_vec_list = [self.embeddings.embed_query(q) for q in querys]
        query_vector = np.array(query_vec_list).astype('float32')
        faiss.normalize_L2(query_vector)
        
        distances, indices = self.index.search(query_vector, k)
        return indices, distances

if __name__ == "__main__":
    import os
    from config import get_config
    config = get_config()

    embeddings = DocProcessingEmbeddings.from_config(config)

    from langchain_chroma import Chroma
    host = config.get("chroma_server_host")
    if host:
        chroma = Chroma(
            collection_name="lotus",
            embedding_function=embeddings,
            host=host,
            port=int(config.get("chroma_server_port", 8000)),
            relevance_score_fn="l2",
        )
    else:
        chroma = Chroma(
            collection_name="lotus",
            embedding_function=embeddings,
            persist_directory=os.path.join(config["persist_directory"], "chroma"),
            relevance_score_fn="l2",
        )

    docs = chroma.get(include=["metadatas", "embeddings"])
    retriever = FaissRetriever(docs['embeddings'], embeddings)

    querys = ["Lotus Technology Company (LTC) was incorporated as an exempted company in accordance with the laws and regulations of the Cayman Islands on August 9, 2021. The mailing address of Lotus Technology's principal executive office is No. 800 Century Avenue, Pudong District, Shanghai, People’s Republic of China, and the phone number is +86 21 5466 - 6258. Lotus Technology's corporate website address is www.group-lotus.com. The information contained in, or accessible through, Lotus Technology's website does not constitute a part of this prospectus."]

    indices, distances = retriever.invoke(querys, 1000)

    # save indices and distances into a log file
    with open("faiss_retriever2.log", "w") as f:
        f.write(f"indices: {indices}\n")
        f.write(f"distances: {distances}\n")
