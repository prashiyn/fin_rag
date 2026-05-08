import sys
import os
import time
import logging
import json
logging.basicConfig(
    filemode='w',
    filename='chat_service_test.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.vllmChatService import ChatService
from src.utils.ragManager import RAGManager
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
        print(f"Error：{e}")
    return questions

def write_wrapped_text(file, text, max_line_length=80):
    while text:
        # 写入每行最多 max_line_length 个字符
        file.write(text[:max_line_length] + '\n')
        text = text[max_line_length:]


def load_from_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    
    config = get_config()
    
    collections = {'lotus': 10, 'lotus_car_stats': 0, 'lotus_brand_info': 0}
    rag_manager = RAGManager(config=config, collections=collections)
    chat_service = ChatService(config=config, rag_manager=rag_manager)
    
    data = load_from_json('109_testingset.json')

    for idx, entry in enumerate(data):
        question = entry.get("question")
        print(f"Question {idx + 1}: {question}")
        session_id = "test"
        chat_manager = chat_service.get_or_create_chat_manager(session_id)
        rewritten = chat_manager.if_query_rag(question, "", 5)
        entry["rewritten"] = rewritten
    
    with open('109_testingset_rewritten_list.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
