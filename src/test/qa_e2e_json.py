import sys
import os
import time
import logging
import json
from datetime import datetime
import uuid
import re 
import torch


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
script_dir = os.path.dirname(os.path.abspath(__file__))

from utils.vllmChatService import ChatService
from utils.ragManager import RAGManager
from gpu_log import log_gpu_usage
from config import get_config


log_dir = os.path.join(script_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'qa_e2e_json_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
logging.basicConfig(
    filemode='w',
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    
    log_gpu_usage("Test Start")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    logger = logging.getLogger(__name__)
    
    
    config = get_config()

    print("Reranker model: ", config['rerank_model'])

    collections = {'lotus': 10}
    collection_name = "lotus"
    logger.warning("Before loading: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))

    rag_manager = RAGManager(config=config, collections=collections)
    logger.warning("Load retriever: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))

    chat_service = ChatService(config=config, rag_manager=rag_manager, rerank_topk = config['rerank_topk'])
    logger.warning("Load Reranker: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))

    questions_folder_path = os.path.join(script_dir, './test_questions')
    results_folder_path = os.path.join(script_dir, './test_results/1')
    os.makedirs(results_folder_path, exist_ok=True)

    
    questions_files = [f for f in os.listdir(questions_folder_path) if os.path.isfile(os.path.join(questions_folder_path, f)) and f.endswith('.json')]
    questions_files = ["single_test.json"] # specify the question file(s) or iterate all files under /test_questions/
    
    session_id = str(uuid.uuid4())

    for i in range(len(questions_files)):  
        question_file = questions_files[i] 
        questions_file_path = os.path.join(questions_folder_path, question_file)
        
        # Extract batch number from filename
        batch_match = re.search(r'batch(\d+)', question_file)
        batch_num = batch_match.group(1) if batch_match else "0"
        
        # Special cases
        if not batch_match and question_file == "single_test.json":
            batch_num = "single_test3"
        if not batch_match and question_file == "multi_subquestion.json":
            batch_num = "multi_subquestion"

        with open(questions_file_path, 'r', encoding='utf-8') as f:
            question_data = json.load(f)

        for idx, item in enumerate(question_data):
            question = item['question']
            print(f"Processing: {question}")
            
        
            (answer, rag_context, rag_info, rewritten_question, hypo_chunk_content, 
             all_retrieved_content, qa_history
            ) = chat_service.generate_response_with_rag(
                question, session_id, collection_name, internal_input=None, interrupt_index=None)
            
            chat_manager = chat_service.get_or_create_chat_manager(session_id, collection_name)

            simplified_content = []

            for each_sub_query in all_retrieved_content:
                sub_query_content = []
                for item in each_sub_query:
                    # Extract only the needed properties
                    simplified_item = {
                        "retriever": item["retriever"],
                        "score": item["score"],
                        "page_content": item["page_content"],
                        "date_published": item["metadata"]["date_published"]
                    }
                    sub_query_content.append(simplified_item)
                simplified_content.append(sub_query_content)
            
            hypo_content = hypo_chunk_content[0] if hypo_chunk_content else None

            result_data = {
                "question": question,
                "rewritten": rewritten_question if rewritten_question else [],
                "rag": chat_manager.need_rag,
                "answer": answer,
                "history": qa_history,
                "hyde": hypo_content,
                "reranked": rag_info['chunk_content'].tolist() if not rag_info.empty else [],
                "all_retrieved": simplified_content
            }



            output_filename = f"{batch_num}_{idx+1}.json"
            output_json_path = os.path.join(results_folder_path, output_filename)
            
            # file_write_start_time = time.perf_counter()

            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)

            # file_write_end_time = time.perf_counter()
            # file_write_duration = file_write_end_time - file_write_start_time
            # logger.info(f"The time for writing question entry [{question}] to {output_filename}: {file_write_duration:.2f} seconds")
            
        print("-"*50)
        print(f"Processed batch {batch_num} to: {results_folder_path}, at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # keep QA history for the whole test
    chat_service.get_or_create_chat_manager(session_id, collection_name).clear_chat_history()