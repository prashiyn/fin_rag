import os
import numpy as np

try:
    from src.services.doc_processing_llm import DocProcessingEmbeddings
    from src.config import get_config
except ImportError:
    from services.doc_processing_llm import DocProcessingEmbeddings
    from config import get_config

# This model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
# They are defined in `config_sentence_transformers.json`
query_prompt_name = "s2p_query"
queries = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]
# docs do not need any prompts
docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]

cfg = get_config()
emb = DocProcessingEmbeddings.from_config(cfg)
query_embeddings = np.array(emb.embed_documents(queries), dtype=np.float32)
doc_embeddings = np.array(emb.embed_documents(docs), dtype=np.float32)
print(query_embeddings.shape, doc_embeddings.shape)

q_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
d_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
similarities = np.matmul(q_norm, d_norm.T)
print(similarities)
