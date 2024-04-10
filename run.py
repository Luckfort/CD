from pathlib import Path
from tqdm.notebook import tqdm
import multiprocessing
import warnings
import gc
import torch
import os
import time
import random

# This is the script to run the OpenKE model in parallel;

script = "main.py"

parellel = 1

def run_model(task):
    task_id, model_name, data_name, quant, noise = task[0], task[1], task[2], task[3], task[4]
    print(f"Running {script}: Data: {data_name}, Model: {model_name}, Quantization: {quant}-bit")
    #device = int(task_id % (torch.cuda.device_count()))
    
    time.sleep(random.randint(0, 5)) # sleep for a while to avoid CPU overload
    comment = f"python3 {script} --model {model_name} --quant {quant} --cuda {0} --noise {noise} --dataset {data_name} --clf LR"
    print(comment)
    os.system(comment)
    gc.collect()

"""
    Run LLM
"""
def run_LLM():
    task_id, tasks = 0, []
    data_name_list = ['StrategyQA','coinflip','cities','common','counterfact','STSA','IMDb','sarcasm','hateeval']
    
    model_name_list = ["google/gemma-2b", "google/gemma-7b", 
                       "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf",
                       "Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-1.8B", 
                       "Qwen/Qwen1.5-4B","Qwen/Qwen1.5-7B", "Qwen/Qwen1.5-14B"]
    
    for model_name in model_name_list:
        for data_name in data_name_list:
            for noise in ['non-noise','noise']:
                for quant in [32, 16, 8]:
                    tasks.append([task_id, model_name, data_name, quant, noise])
                    task_id += 1
    with multiprocessing.Pool(parellel) as p:
        p.map(run_model, tasks)

def run_all_models():
    run_LLM()

run_all_models()