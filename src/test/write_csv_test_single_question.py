# RAG自动填表单题或多题测试脚本

import sys
import os
import time
import logging
import json
from pathlib import Path
import csv
import shutil
import uuid
import requests
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.vllmChatService import ChatService
from utils.ragManager import RAGManager
from services.doc_processing_llm import DocProcessingLLMClient
from config import get_config

# -------------csv_path-----------------
csv_path = "/root/autodl-tmp/dir_tzh/lotus_dataset/filled.csv"
output_csv = "/root/autodl-tmp/dir_tzh/lotus_dataset/filled.csv"

# ---------- 把列表写成 JSON ----------
out_json_file = Path("/root/autodl-tmp/dir_tzh/lotus_dataset/write_csv_json/4.json")
all_retrieved_json_path = "/root/autodl-tmp/dir_tzh/lotus_dataset/write_csv_json/4.json"
all_retrieved_json = Path(all_retrieved_json_path)

# 复制文件
if csv_path != output_csv:
    shutil.copy(csv_path, output_csv)

# # csv中要包含主题字段
# FIXED_COLS  = ["topic"]
# # 1. 取首行字段
# with open(csv_path, newline="", encoding="utf-8") as f:
#     header = next(csv.reader(f))          # header -> list[str]

# # 2. 过滤掉固定列
# period_columns = [col.strip() for col in header if col.strip() not in FIXED_COLS]

# print(period_columns)


def _load_config() -> dict:
    return get_config()


_cfg = _load_config()
MODEL_NAME = _cfg["test_llm_model_name"]
client = DocProcessingLLMClient.from_config(_cfg)


def generate_question(period_code,topic, max_retry=2):
    prompt = f"""
    你是一位针对路特斯公司的中文问题生成器。请根据下列输入，**仅输出一句自然、简洁的疑问句**，不要包含任何解释或多余内容。

    —— 输入字段 ——
    时间码: {period_code}
    主题: {topic}

    —— 时间码释义规则 (务必遵守) ——
    • 代码格式：Y<年份>_<标签>  
    - Q1 → “一季度”  
    - Q2 → “二季度”  
    - Q3 → “三季度”  
    - Q4 → “四季度”  
    - H1 → “上半年”  
    - H2 → “下半年”  
    - FY → “全年”   
    例如：Y2024_Q1 = “2024年一季度”，Y2025_FY = “2025年全年”。

    —— 生成要求 ——
    1. 先把时间码按以上规则转换成中文自然描述。  
    2. 输出句式固定为：“<公司><时间描述>的<主题>是多少？”  
    3. 只输出这一句话，不要多余标点、前后缀或换行。

    请根据上述要求生成疑问句。

    例如：主题为销量 时间码是Y2024_Q1 那么问题应该是：路特斯2024年一季度的销量是多少？
    """


    for _ in range(max_retry):

        try:
            completion = client.complete(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一位针对路特斯公司的中文问题生成器"},
                    {"role": "user", "content": prompt}
                ],
            )
            question = str(completion.get("content", "")).strip()
            break
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            continue

    return question    
        


config = get_config()

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

logger = logging.getLogger(__name__)

collections = {'lotus': 15}
collection_name = "lotus"
rag_manager = RAGManager(config=config, collections=collections)
chat_service = ChatService(config=config, rag_manager=rag_manager, rerank_topk=config['rerank_topk'])

session_id = str(uuid.uuid4())

def rag_answer(question):
    answer, rag_context, rag_info, rewritten_question, hypo_chunk_content, all_retrieved_content, history = chat_service.generate_response_async(
        question,
        session_id,
        collection_name,
        internal_input=None,
        interrupt_index=None,
        using_qa_pairs=False
    )
    chat_service.get_or_create_chat_manager(session_id, collection_name).clear_chat_history()
    return answer, rag_context, rag_info, rewritten_question, all_retrieved_content, history

# 判断答案 并简化答案为一个数字
def judge_answer(question,answer,max_retry=2):
    # prompt = f"""
    # 你是一名数据提炼助手。  
    # 给定一条关于「路特斯公司」的 **问题** 和对应的 **完整回答**，请根据问题所关注的核心量化信息，将回答**简化**为 “唯一的数字 + 单位” 的形式，输出时只保留这一串文本，禁止添加任何多余字符（如句号、空格、引号或解释说明）。

    # - 如果回答中包含多个数字，请判断哪个数字最能直接回答问题并只输出它及其单位。  
    # - 若回答未提及数字或单位，返回空字符串。  

    # —— 输入字段 ——
    # 问题: {question}
    # 答案: {answer}

    # - 输出示例：  
    #     - `3000辆`  
    #     - `5.6亿美元`
    
    # """
    prompt = f"""
        你是一名“问答校验与数字提炼”助手。  
        系统会一次性提供：

        问题：{question}  
        回答：{answer}

        请依下列规则输出 **两行文本**：

        ──────────────────────────────────────────
        第 1 行：  
        - 输出 `True`，如果回答 **成功** 回应了问题的核心（即回答中确实给出了能直接解答问题的具体数字信息）。  
        - 否则输出 `False`。判断为 False 的典型情形包括但不限于：  
        • 回答出现 “数据尚未公开披露”“暂无公开数据”“无法评估”“暂无确切数字” 等表述；  
        • 回答里根本没有出现任何数字；  
        • 回答与问题不相符或答非所问。

        第 2 行：  
        - 当第 1 行为 `True` 时，提炼回答中 **最能直接回答问题** 的 “唯一数字 + 单位” 片段，并原样输出；禁止出现多余字符（如空格、标点、解释）。
        - 若回答含多个数字，请选取能一锤定音回答问题的那一个。  
        - 示例输出：  
            - `3000辆`  
            - `5.6亿美元`
        - 当第 1 行为 `False` 时，输出固定字符串 `N/A`。
        ──────────────────────────────────────────

        ⚠️ 绝不能在任一行添加多余的空格、标点或说明性文字。
    """
    
    for _ in range(max_retry):
        try:
            completion = client.complete(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一名数据提炼助手。"},
                    {"role": "user", "content": prompt}
                ],
            )
            response_lines = str(completion.get("content", "")).strip().split("\n")
            success_str = response_lines[0].strip()
            success = success_str.lower() == 'true'      # True 或 False (bool)
            answer = response_lines[1].strip()
            break
        except Exception as e:
            print(f"API调用错误: {str(e)}")
            continue
    return success, answer    
    
def R1_online_to_sse_text(question):
    url = "https://wss.lke.cloud.tencent.com/v1/qbot/chat/sse"
    payload = {
        "session_id":      "a29bae68-cb1c-489d-8097",
        "bot_app_key":     "ZqNNJJpfaTUAMYADzVLWoqvYCukTREWWQegWlebfPCbwWdiizsqCJoCKNDMrWIDgBpQyLDELluvxwSBjZXvtmSEeDTfKYUuDdLHwARrTJTtjUmxJHkDUPZzLkCfNTrOK",
        "visitor_biz_id":  "a29bae68-cb1c-489d-8097",
        "content":         question,
        "incremental":     True,
        "streaming_throttle": 10,
        "visitor_labels":  [],
        "custom_variables": {},
        "search_network":"enable"
    }


    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        stream=False, timeout=None
    )
    resp.encoding = "utf-8"          # 告诉 requests 按 UTF‑8 解码
    sse_text = resp.text             # 现在就是正常中文
    # print(sse_text)
    return sse_text



def extract_reply_contents(sse_text: str) -> list[str]:
    """
    从完整的 SSE 文本中提取所有 event:reply 的 payload.content。
    返回一个列表，按出现顺序排列。
    """
    contents = []
    current_event = None
    data_buffer = []

    for line in sse_text.splitlines():
        if line.startswith("event:"):
            # 遇到新事件——先把上一段 data 处理掉
            if current_event == "reply" and data_buffer:
                data_json = json.loads("".join(data_buffer))
                contents.append(data_json["payload"]["content"])
            # 重置并记录新事件名
            current_event = line[6:].strip()
            data_buffer = []
        elif line.startswith("data:"):
            # 去掉开头 "data:" 累积 JSON 字符串
            data_buffer.append(line[5:].strip())

    # 处理文本末尾最后一段（若也是 reply）
    if current_event == "reply" and data_buffer:
        data_json = json.loads("".join(data_buffer))
        contents.append(data_json["payload"]["content"])

    return contents

# question = ["路特斯2023年一季度销量","路特斯2023年二季度销量"]


# for q in question:
#     answer, _, rag_info, rewritten_question, all_retrieved_content, _ = rag_answer(q)
    
#     success, answer = judge_answer(q,answer)
#     print(type(success))
#     print(success)
#     print(answer)

out_json_file = Path("/root/autodl-tmp/hyc_production/RAG_Agent/src/test/write_csv_test.json")
# questions = ["路特斯2023年一季度研发费用","路特斯2023年二季度研发费用",
#              "路特斯2023年一季度现金余额","路特斯2023年二季度现金余额",
#              "路特斯2023年一季度销售收入","路特斯2023年一季度商品销售收入"
#              ,"路特斯2024年一季度商品销售收入","路特斯2024年一季度服务销售收入",
#              "路特斯2024年二季度商品销售收入","路特斯2024年二季度服务销售收入",
#              "路特斯2024年上半年服务销售收入","路特斯2024年下半年毛利率",
#              "路特斯2024年四季度中国门店数量","路特斯2024年年底中国门店数量"]

questions = ["路特斯2023年一季度的销量是多少？"]
results = []
for q in questions:
    answer_rag, _, rag_info, rewritten_question, all_retrieved_content, _ = rag_answer(q)        
    success,answer = judge_answer(q,answer_rag)

                        
    print(q)
    print(rewritten_question)
    print(answer)
    print(rag_info)
    results.append({
            "question"    : q,
            "rewritten"   : rewritten_question,
            "source"      : "RAG",
            "answer"      : answer,
            "complete_answer" : answer_rag,
            "rag_info"    : rag_info.to_json(orient='records'),
            "all_retrieved"    : all_retrieved_content
            })

with out_json_file.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)                 



