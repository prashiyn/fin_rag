import sys
import os
import time
import logging
import json
import glob
import torch
logging.basicConfig(
    filemode='w',
    filename='profile_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.vllmChatService import ChatService
from src.utils.ragManager import RAGManager
from src.utils.profiler import profiler
from src.config import get_config

def read_questions_from_md(md_file_path):
    questions = []
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                question = line.strip()
                if question:
                    questions.append(question)
    except FileNotFoundError:
        print(f"'{md_file_path}' not found")
    except Exception as e:
        print(f"Error: {e}")
    return questions

def load_questions_file(file_path):
    """Load questions from either JSON or text files"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension in ['.txt', '.md']:
            # Load text file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                # Convert to same format as JSON data
                data = [{'question': line, 'answer': ''} for line in lines]
        elif file_extension == '.json':
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        return data
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {str(e)}")
        raise

if __name__ == "__main__":
    
    config = get_config()
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    logger = logging.getLogger(__name__)

    collections = {'lotus': 10}
    logger.warning("Before loading: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))
    rag_manager = RAGManager(config=config, collections=collections)
    logger.warning("Load retriever: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))
    chat_service = ChatService(config=config, rag_manager=rag_manager, rerank_topk=5)
    logger.warning("Load Reranker: Max CUDA memory allocated: {} GB".format(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)))
    
    subquestions_dir = "../../src/test/75_question/"
    subquestion_files = glob.glob(os.path.join(subquestions_dir, "*.json"))
    
    if not subquestion_files:
        logger.error(f"No markdown files found in {subquestions_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(subquestion_files)} subquestion files: {[os.path.basename(f) for f in subquestion_files]}")
    
    BATCH_SIZE = 1
    
    for questions_file_path in subquestion_files:
        base_name = os.path.splitext(os.path.basename(questions_file_path))[0]
        logger.info(f"Processing dataset: {base_name}")
        if base_name == "1":
            continue
        
        output_dir = os.path.join(os.path.dirname(questions_file_path), f'results_{base_name}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        try:
            # data = load_questions_file(questions_file_path)
            with open(questions_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            num_questions = len(data)
            logger.info(f"Loaded {num_questions} questions from {base_name}")
            profile_output_file = os.path.join(output_dir, f"profile_{base_name}.json")
            
            for i in range(0, num_questions, BATCH_SIZE):
                batch_questions = data[i:i+BATCH_SIZE]
                
                session_id = time.time()
                for idx, item in enumerate(batch_questions):
                    logger.info(f"Processing question {i+idx+1}/{num_questions}")
                    question = item['question']
                    
                    profiler.start('response')
                    generator = chat_service.generate_response_async_stream(
                        question, session_id, internal_input=None, interrupt_index=None
                    )
                    
                    try:
                        first_yield = next(generator)
                    except StopIteration:
                        logger.error(f"Generator did not yield any values for question: {question}")
                    
                    profiler.end('response')

                    profiler.log_profiling_results(profile_output_file)

                    time.sleep(8)
            
            
            
        except Exception as e:
            logger.error(f"Error processing {base_name}: {str(e)}")
            continue
    
    logger.info("Completed profiling for all subquestion datasets")
