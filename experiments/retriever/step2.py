'''
Step 2:
    1. Recall from rewritten questions (and hypo chunks)
    2. Get recall metrics (Recall, Precision, F1 ...)

Input
    Json file with the following structure:
    [
        {
            "question": "original question",
            "rewritten": ["rewritten question 1", "rewritten question 2"],
            "hyde": ["hyde 1", "hyde 2"] (optional),
            "perplexity": 0.0 (optional)
        }
        ...
    ]

Output
    Json file with the following structure:
    [
        {
            "avg_recall": 0.0,
            "avg_precision": 0.0,
            "avg_f1": 0.0,
            "avg_ppl": 0.0 (-1 if no perplexity provided),
        }
        {
            "question": "original question",
            "rewritten": ["rewritten question 1", "rewritten question 2"],
            "hyde_ppl": 0.0 (if no perplexity provided, default to -1),
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0
        }
        ...
    ]
'''

import os
import json
import logging
import argparse
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from langchain_chroma import Chroma
from src.services.doc_processing_llm import DocProcessingEmbeddings
from src.config import get_config

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.ensembleRetriever import EnsembleRetriever

logging.basicConfig(
    filemode='w',
    filename='step2.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

def load_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json_file(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def evaluate_recall(file_path, answer_file, retriever, args):
    data = load_json_file(file_path)
    answer_data = load_json_file(answer_file)
    results = []

    for idx, (entry, answer) in tqdm(enumerate(zip(data, answer_data))):
        try:
            rewritten = entry.get("rewritten", [])
            hyde = entry.get("hyde", [])
            query_chunks = []

            if isinstance(rewritten, str):
                rewritten = [rewritten]

            if args.enable_hyde:
                for q, h in zip(rewritten, hyde):
                    chunks = retriever.invoke(q, h)
                    query_chunks += [chunk['page_content'].strip() for chunk in chunks]
            else:
                for q in rewritten:
                    chunks = retriever.invoke(q, [])
                    query_chunks += [chunk['page_content'].strip() for chunk in chunks]

            assert answer.get("question") == entry.get("question"), f"Questions do not match, {answer.get('question')}\n!=\n{entry.get('question')}"
            
            pos_chunks = answer.get("content_list", [])

            recall = 0
            precision = 0
            f1 = 0
            recall_pos_chunks = set(query_chunks).intersection(set(pos_chunks))

            recall = len(recall_pos_chunks) / len(pos_chunks) if pos_chunks else 0
            precision = len(recall_pos_chunks) / len(query_chunks) if query_chunks else 0
            f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0
            
            results.append({
                "question": entry.get("question"),
                "rewritten": rewritten,
                "hyde_ppl": entry.get("perplexity", -1),
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "num_recalls": len(query_chunks),
                "pos_recalls": list(recall_pos_chunks),
            })
            
        except Exception as e:
            logging.error(f"Error processing entry {idx}: {e}")
            logging.error(f"Entry: {entry}")
            logging.error(f"Answer: {answer}")

    # Calculate averages
    if len(results):
        avg_recall = sum(r['recall'] for r in results) / len(results)
        avg_precision = sum(r['precision'] for r in results) / len(results)
        avg_f1 = sum(r['f1'] for r in results) / len(results)
        avg_ppl = sum((sum(r['hyde_ppl']) / len(r['hyde_ppl'])) if isinstance(r['hyde_ppl'], list) else r['hyde_ppl'] for r in results) / len(results)
        
        overall = {
            'avg_recall': avg_recall,
            'avg_precision': avg_precision,
            'avg_f1': avg_f1,
            'avg_ppl': avg_ppl,
            'avg_num_recalls': sum(r['num_recalls'] for r in results) / len(results),
        }
        results.insert(0, overall)

    return results

def save_visualization(results, output_path):
    precisions = [r['precision'] for r in results[1:]]
    recalls = [r['recall'] for r in results[1:]]
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(precisions, bins=20)
    plt.title('Precision Distribution')
    plt.xlabel('Precision')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.hist(recalls, bins=20)
    plt.title('Recall Distribution')
    plt.xlabel('Recall')
    plt.ylabel('Count')

    plt.tight_layout()
    
    # Save visualization next to the output JSON
    png_path = os.path.splitext(output_path)[0] + '.png'
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate rewritten questions using hyde models')
    parser.add_argument('--input', type=str, default="hyde.json", help='Input JSON file path')
    parser.add_argument('--output', type=str, default="recall_metric/", help='Output dir path')
    parser.add_argument('--answer', type=str, default='/root/autodl-tmp/RAG_Agent_vllm_cjj/eval/answer/75.json', help='Answer file path')
    parser.add_argument('--faiss_k', type=int, default=40, help='Faiss topk')
    parser.add_argument('--bm25_k', type=int, default=10, help='BM25 topk')
    parser.add_argument('--faiss_ts_k', type=int, default=10, help='Faiss topk for title and snippet')
    parser.add_argument('--enable_expand', action='store_true', help='Expand chunk content')
    parser.add_argument('--enable_hyde', action='store_true', help='Enable Hyde')
    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Load configuration
    config = get_config()

    # Initialize retriever
    collection_name = "lotus"
    embeddings = DocProcessingEmbeddings.from_config(config)
    host = config.get("chroma_server_host")
    port = int(config.get("chroma_server_port", 8000))

    if host:
        chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            host=host,
            port=port,
            relevance_score_fn="l2",
        )
        ts_chroma = Chroma(
            collection_name=f"{collection_name}_ts",
            embedding_function=embeddings,
            host=host,
            port=port,
            relevance_score_fn="l2",
        )
    else:
        chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=os.path.join(config["persist_directory"], "chroma"),
            relevance_score_fn="l2",
        )
        ts_chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=os.path.join(config["persist_directory"], "ts_chroma"),
            relevance_score_fn="l2",
        )

    bm25_dir = os.path.join(config["persist_directory"], "bm25_index", collection_name)
    retriever = EnsembleRetriever(
        bm25_dir, 
        chroma,
        ts_chroma,
        10,
        embeddings,
        faiss_k=args.faiss_k,
        bm25_k=args.bm25_k,
        faiss_ts_k=args.faiss_ts_k,
        enable_expand=args.enable_expand,
    )

    # Run evaluation
    results = evaluate_recall(args.input, args.answer, retriever, args)
    
    # Save results

    # Ensure the output directory exists
    os.makedirs(args.output, exist_ok=True)
    file_path = os.path.join(args.output, "result_2.json")
    save_json_file(results, file_path)
    logging.info(f"Evaluation results saved to {file_path}")
    
    # Generate and save visualization
    png_path = os.path.join(args.output, "result_2_metric.png")
    save_visualization(results, png_path)
    logging.info(f"Visualization saved to {png_path}")

if __name__ == "__main__":
    main()
