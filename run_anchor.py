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
    model_name_log = model_name.replace('/','-')
    log_name = f"parallel_gpu_{model_name_log}_{data_name}.log"
    
    time.sleep(random.randint(0, 5)) # sleep for a while to avoid CPU overload
    comment = f"python3 {script} --model {model_name} --quant {quant} --noise {noise} --dataset {data_name} --clf LR > {log_name}"
    print(comment)
    os.system(comment)
    gc.collect()

"""
    Run LLM
"""
def run_LLM():
    task_id, tasks = 0, []
    data_name_list = ['cities','common','counterfact','STSA','IMDb','sarcasm','hateeval','StrategyQA','coinflip']
    model_name_list = ["meta-llama/Meta-Llama-3-8B-Instruct"]
    
    for model_name in model_name_list:
        for data_name in data_name_list:
            for noise in ['non-noise']:
                for quant in [32]:
                    tasks.append([task_id, model_name, data_name, quant, noise])
                    task_id += 1
    with multiprocessing.Pool(parellel) as p:
        p.map(run_model, tasks)

def run_all_models():
    run_LLM()

run_all_models()