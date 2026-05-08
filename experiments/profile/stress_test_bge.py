import torch
import threading
import time
import random
from collections import deque
from datetime import datetime
import json
import sys
import os
from FlagEmbedding import FlagLLMReranker

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.ragManager import RAGManager
from src.utils.vllmChatService import ChatService
from src.config import get_config

question_file = "/root/autodl-tmp/cjj/RAG_Agent/src/test/test_questions/question_batch1.md"
with open(question_file, 'r', encoding='utf-8') as f:
    questions = [q.strip() for q in f.readlines() if q.strip()]

# A list of random chunks in JSON list
chunks_file = "chunks.json"
with open(chunks_file, 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# Global statistics variables
class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_inference_calls = 0
        self.inference_calls_per_model = {}
        self.recent_inference_calls = deque(maxlen=100)  # Store recent timestamps for throughput calculation
        self.start_time = None  # Will be set after warm-up period
        self.warm_up_complete = False
        self.warm_up_start_time = time.time()
        self.warm_up_period = 30  # 30 seconds warm-up period
    
    def add_inference_call(self, model_name):
        with self.lock:
            # Check if we're still in warm-up period
            current_time = time.time()
            if not self.warm_up_complete:
                if current_time - self.warm_up_start_time >= self.warm_up_period:
                    # Warm-up period is over, start counting from now
                    self.warm_up_complete = True
                    self.start_time = current_time
                    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Warm-up period complete. Starting to collect metrics.")
                else:
                    # Still in warm-up, don't count this call
                    return
            
            # Count this call since warm-up is complete
            self.total_inference_calls += 1
            if model_name not in self.inference_calls_per_model:
                self.inference_calls_per_model[model_name] = 0
            self.inference_calls_per_model[model_name] += 1
            self.recent_inference_calls.append((current_time, 1))
    
    def get_stats(self):
        with self.lock:
            # If we're still in warm-up period, show that
            if not self.warm_up_complete:
                current_time = time.time()
                warm_up_elapsed = current_time - self.warm_up_start_time
                remaining = max(0, self.warm_up_period - warm_up_elapsed)
                return {
                    "in_warm_up": True,
                    "warm_up_remaining": remaining,
                    "total_inference_calls": 0,
                    "inference_calls_per_model": {},
                    "overall_rate": 0,
                    "recent_rate": 0,
                    "elapsed_time": 0
                }
            
            # Calculate stats after warm-up period
            elapsed = time.time() - self.start_time if self.start_time else 0
            overall_rate = self.total_inference_calls / elapsed if elapsed > 0 else 0
            
            # Calculate recent throughput (last 100 inference calls)
            recent_rate = 0
            if self.recent_inference_calls:
                oldest_time = self.recent_inference_calls[0][0]
                newest_time = self.recent_inference_calls[-1][0]
                recent_calls = sum(calls for _, calls in self.recent_inference_calls)
                recent_elapsed = newest_time - oldest_time
                if recent_elapsed > 0:
                    recent_rate = recent_calls / recent_elapsed
            
            return {
                "in_warm_up": False,
                "total_inference_calls": self.total_inference_calls,
                "inference_calls_per_model": dict(self.inference_calls_per_model),
                "overall_rate": overall_rate,
                "recent_rate": recent_rate,
                "elapsed_time": elapsed
            }

# Initialize global stats
stats = Stats()

# Model configurations - all using GPU
MODEL_CONFIGS = [
    {'name': 'model1', 'model_id': '/root/autodl-tmp/cjj/cache/models--BAAI--bge-reranker-v2-gemma-1/models--BAAI--bge-reranker-v2-gemma-1/snapshots/1787044f8b6fb740a9de4557c3a12377f84d9e18'},
    {'name': 'model2', 'model_id': '/root/autodl-tmp/cjj/cache/models--BAAI--bge-reranker-v2-gemma-2/models--BAAI--bge-reranker-v2-gemma-2/snapshots/1787044f8b6fb740a9de4557c3a12377f84d9e17'},
    {'name': 'model3', 'model_id': '/root/autodl-tmp/cjj/cache/models--BAAI--bge-reranker-v2-gemma-3/models--BAAI--bge-reranker-v2-gemma-3/snapshots/1787044f8b6fb740a9de4557c3a12377f84d9e16'},
    {'name': 'model4', 'model_id': '/root/autodl-tmp/cjj/cache/models--BAAI--bge-reranker-v2-gemma-4/models--BAAI--bge-reranker-v2-gemma-4/snapshots/1787044f8b6fb740a9de4557c3a12377f84d9e15'}
    
]

def print_stats():
    """Print current statistics"""
    while True:
        current_stats = stats.get_stats()
        
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Performance Statistics:")
        
        # Check if we're still in warm-up period
        if current_stats.get('in_warm_up', False):
            print(f"WARM-UP PERIOD: {current_stats['warm_up_remaining']:.1f} seconds remaining")
            print("Metrics will start being collected after warm-up period completes.")
        else:
            # Regular stats display after warm-up
            print(f"Total inference calls: {current_stats['total_inference_calls']}")
            print(f"Elapsed time: {current_stats['elapsed_time']:.2f} seconds")
            print(f"Overall rate: {current_stats['overall_rate']:.2f} inference calls/second")
            print(f"Recent rate: {current_stats['recent_rate']:.2f} inference calls/second")
            print("Inference calls per model:")
            for model, count in current_stats['inference_calls_per_model'].items():
                print(f"  {model}: {count}")
        
        time.sleep(10)  # Update stats every 10 seconds

def model_worker(config):
    """Worker function to run a model in a separate thread"""

    model_name = config['name']
    model_id = config['model_id']
    reranker = config['reranker']
    retriever = config['retriever']
    chat_manager = config['chat_manager']
    device = config.get('device', 'cuda:0')
    
    print(f"{model_name} initialized and ready on {device}")

    # Run inference in an infinite loop
    while True:
        try:

            question = random.choice(questions)
            k = random.randint(50, 180)
            tmp_chunks = random.choices(chunks, k=k)

            tmp_chunks = [{"page_content": chunk,
                           "metadata": {"date_published": "2023-01-01"},
                           "bundle_id": i} for i, chunk in enumerate(tmp_chunks)]
            print(f"{model_name} processing {len(tmp_chunks)} chunks")
            _ = chat_manager.rank_chunk(tmp_chunks, question, datetime.now(), retriever, reranker)
            print(f"{model_name} processed {len(tmp_chunks)} chunks")
            
            stats.add_inference_call(model_name)
            
            torch.cuda.empty_cache()

            print(f"{model_name} processed {len(tmp_chunks)} chunks")
            
        except Exception as e:
            print(f"Error in {model_name}: {str(e)}")
            time.sleep(1)  # Prevent tight loop in case of recurring errors

# Start the stats printing thread
stats_thread = threading.Thread(target=print_stats, daemon=True)
stats_thread.start()

rag_config = get_config()

collections = {'lotus': 10}
rag_manager = RAGManager(config=rag_config, collections=collections)
rag_manager.create_collection("lotus")


# Pre-load all models to GPU
print("Pre-loading all models to GPU...")
device = 'cuda:0'
for config in MODEL_CONFIGS:
    model_name = config['name']
    model_id = config['model_id']
    print(f"Loading {model_name} with {model_id} to {device}...")
    
    config['reranker'] = FlagLLMReranker(
        model_id,
        use_fp16=True,
        # cache_dir=config['cache_dir'],
        devices=[device]
    )
    config['retriever'] = rag_manager.create_retriever(10, "lotus", retriever_type="ensemble")
    chat_service = ChatService(rag_config, None, 5)
    config['chat_manager'] = chat_service.get_or_create_chat_manager(config['name'])
    print(f"{model_name} loaded successfully")

# Start model worker threads
model_threads = []
for config in MODEL_CONFIGS:
    thread = threading.Thread(target=model_worker, args=(config,), daemon=True)
    thread.start()
    model_threads.append(thread)
    time.sleep(0.5)  # Small delay between thread starts

# Function to save stats to a file
def save_stats_to_file(filename):
    current_stats = stats.get_stats()
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Format the stats as a readable text file
    with open(filename, 'w') as f:
        f.write(f"Test Reranker Statistics - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Runtime: {current_stats['elapsed_time']:.2f} seconds\n")
        f.write(f"Total Inference Calls: {current_stats['total_inference_calls']}\n")
        f.write(f"Overall Rate: {current_stats['overall_rate']:.2f} inference calls/second\n")
        f.write(f"Recent Rate: {current_stats['recent_rate']:.2f} inference calls/second\n\n")
        
        f.write("Inference Calls Per Model:\n")
        for model, count in current_stats['inference_calls_per_model'].items():
            f.write(f"  {model}: {count}\n")
    
    print(f"\nFinal statistics saved to {filename}")

# Run for exactly 20 minutes (1200 seconds)
print(f"\nTest will run for 20 minutes and then save results to 'reranker_stats.txt'")
start_time = time.time()
test_duration = 600  # 20 minutes in seconds

try:
    while time.time() - start_time < test_duration:
        time.sleep(1)
    
    print("\nTest completed after 20 minutes.")
    # Save the final stats to a file
    save_stats_to_file(f"stress_stats_bge_{len(MODEL_CONFIGS)}models.txt")
    print("Shutting down...")
    
except KeyboardInterrupt:
    print("\nTest interrupted before completion.")
    # Still save whatever stats we have
    save_stats_to_file(f"stress_stats_bge_{len(MODEL_CONFIGS)}models.txt")
    print("Shutting down...")
    
# The daemon threads will be automatically terminated when the main thread exits
