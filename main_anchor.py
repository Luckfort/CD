from dataset_anchor import DataProcessing_Anchor
import json
import argparse
import os
import sys
import torch.nn.functional as F
from collections import OrderedDict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

import torch
#from huggingface_hub import notebook_login
#notebook_login()
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from accelerate import Accelerator
accelerator = Accelerator()

parser = argparse.ArgumentParser()

parser.add_argument('--savepath', type=str, default="./lab_rs/", help="Path for save Result")
parser.add_argument('--model_path', type=str, default=".")
parser.add_argument('--dataset', type=str, default='cities', help="Dataset")
parser.add_argument('--datapath', type=str, default='./dataset/stsa.binary.train', help="Default data path")
parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Which LLM to use")
parser.add_argument('--quant', type=int, default=32, help="Quantization")
parser.add_argument('--noise', type=str, default='non-noise', help="Whether to add noise")

args = parser.parse_args()
if args.quant != 32:
    if args.quant == 8:
        quantization_config1 = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quant == 16:
        pass
    else:
        raise ValueError("We don't have this quantization bit! Please try 8, 16, 32.")

cache_dir = args.model_path
tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir = cache_dir)

if args.quant == 32:
    model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir = cache_dir)
elif args.quant == 8:
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config = quantization_config1, cache_dir = cache_dir)
elif args.quant == 16:
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, cache_dir = cache_dir)
else:
    raise ValueError("We don't have this quantization bit! Please try 8, 16, 32.")

no_split_module_classes = ['LlamaDecoderLayer']
device_map = infer_auto_device_map(
    model, 
    max_memory={0:"0GiB", 1:"8GiB", 2:"16GiB", 3:"8GiB", 4:"8GiB"},
    no_split_module_classes = no_split_module_classes
)

model = dispatch_model(model, device_map=device_map)
file_path = args.datapath

model_2_layer = {"google/gemma-7b": 28, "google/gemma-2b": 18, "meta-llama/Llama-2-7b-chat-hf": 32,
                 "meta-llama/Llama-2-13b-chat-hf": 40, "meta-llama/Llama-2-70b-chat-hf": 80,
                 "Qwen/Qwen1.5-0.5B": 24, "Qwen/Qwen1.5-1.8B": 24, "Qwen/Qwen1.5-4B": 40, 
                 "Qwen/Qwen1.5-7B": 32, "Qwen/Qwen1.5-14B": 40, "Qwen/Qwen1.5-72B": 80,
                 "meta-llama/Meta-Llama-3-8B": 32, "meta-llama/Meta-Llama-3-8B-Instruct": 32,
                 "google/gemma-7b-it": 28, "Qwen/Qwen2-7B-Instruct": 32}
tot_layer = model_2_layer[args.model]

DP = DataProcessing_Anchor(data_path=args.datapath, data_name=args.dataset, noise=args.noise)
p_question, n_question = DP.dispacher()

# Initialize yes/no tokens
token_ = "Ġ"
token_y_list = torch.tensor([tokenizer.convert_tokens_to_ids(token_ + y) for y in ['yes', 'Yes', 'YES']]).long()
token_n_list = torch.tensor([tokenizer.convert_tokens_to_ids(token_ + n) for n in ['no', 'No', 'NO']]).long()

print("Model is training ...")
list_acc = []
list_f1 = []
list_auc = []

rp_questions_dic =[]
rp_log_dic = []
rp_logs = [[] for _ in range(tot_layer)]
rp_questions = [[] for _ in range(tot_layer)]

rp_log_data_list = []
rp_question_data_list = []

pos = 0
tp = 0
tn = 0
fp = 0
fn = 0
fail = 0

main_device = next(model.parameters()).device

for q in tqdm(p_question):
    input_text = DP.get_prompt(q)
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt").to(main_device)
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        prob_distributions = F.softmax(logits, dim=-1)
        max_id = torch.argmax(prob_distributions).unsqueeze(0)
        token_ = tokenizer.convert_ids_to_tokens(max_id)[0]
    
    prob_y = prob_distributions[0, token_y_list].sum()
    prob_n = prob_distributions[0, token_n_list].sum()
    if prob_y >= prob_n:
        pos += 1
        tp += 1
    else:
        fn += 1

for q in tqdm(n_question):
    input_text = DP.get_prompt(q)
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt").to(main_device)
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        prob_distributions = F.softmax(logits, dim=-1)
        max_id = torch.argmax(prob_distributions).unsqueeze(0)
        token_ = tokenizer.convert_ids_to_tokens(max_id)[0]
    
    prob_y = prob_distributions[0, token_y_list].sum()
    prob_n = prob_distributions[0, token_n_list].sum()
    if prob_y < prob_n:
        pos += 1
        tn += 1
    else:
        fp += 1
        
acc = 1.0 * pos / (len(p_question) + len(n_question))
fr = 1.0 * fail / (len(p_question) + len(n_question))
tp = 1.0 * tp / (len(p_question))
fp = 1.0 * fp / (len(p_question))
tn = 1.0 * tn / (len(n_question))
fn = 1.0 * fn / (len(n_question))

print(f"Acc = {acc}, Fail Rate = {fr}")
print(f"TP = {tp}, TN = {tn}")
print(f"FP = {fp}, FN = {fn}")
