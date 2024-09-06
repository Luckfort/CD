from dataset import DataProcessing
import json
from utils import add_noise, LLM
import argparse
import os
import sys
import torch.nn.functional as F
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
#from huggingface_hub import notebook_login
#notebook_login()
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model
from accelerate import Accelerator
accelerator = Accelerator()

parser = argparse.ArgumentParser()

parser.add_argument('--savepath', type=str, default="./lab_rs/", help="Path for save Result")
parser.add_argument('--model_path', type=str, default="/filer/tmp1/hz624/")
parser.add_argument('--dataset', type=str, default='cities', help="Dataset")
parser.add_argument('--datapath', type=str, default='./dataset/stsa.binary.train', help="Default data path")
parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3-8B', help="Which LLM to use")
#parser.add_argument('--model', type=str, default='google/gemma-2b', help="Which LLM to use")
parser.add_argument('--cuda', type=int, default=1, help="Cuda ID")
parser.add_argument('--quant', type=int, default=32, help="Quantization")
parser.add_argument('--noise', type=str, default='non-noise', help="Whether to add noise")
parser.add_argument('--clf', type=str, default='LR', help="Classifier")

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

device_map = infer_auto_device_map(model, max_memory={0:"0GiB", 1:"0GiB", 2:"32GiB", 3:"0GiB", 4:"0GiB"}, no_split_module_classes=["LlamaDecoderLayer"]) #arrakis
print(device_map)
model = dispatch_model(model, device_map=device_map)

#print(model.named_parameters())
#for name, para in model.named_parameters():
#    print(name, para)

file_path = args.datapath

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

model_2_layer = {"google/gemma-7b": 28, "google/gemma-2b": 18, "meta-llama/Llama-2-7b-chat-hf": 32,
                 "meta-llama/Llama-2-13b-chat-hf": 40, "meta-llama/Llama-2-70b-chat-hf": 80,
                 "Qwen/Qwen1.5-0.5B": 24, "Qwen/Qwen1.5-1.8B": 24, "Qwen/Qwen1.5-4B": 40, 
                 "Qwen/Qwen1.5-7B": 32, "Qwen/Qwen1.5-14B": 40, "Qwen/Qwen1.5-72B": 80,
                 "meta-llama/Meta-Llama-3-8B": 32}
tot_layer = model_2_layer[args.model]

DP = DataProcessing(data_path=args.datapath, data_name=args.dataset, noise=args.noise)

# The prompts are different among the datasets!!!
p_question, n_question = DP.dispacher()

token_ = "Ġ"
token_y_list = []
token_n_list = []
for y in ['yes','Yes','YES']:
    token_y_list.append(tokenizer.convert_tokens_to_ids(token_ + y))
for n in ['no','No','NO']:
    token_n_list.append(tokenizer.convert_tokens_to_ids(token_ + n))
token_y_list = torch.Tensor([token_y_list]).long()
token_n_list = torch.Tensor([token_n_list]).long()

Model = LLM(cuda_id = args.cuda, layer_num = tot_layer, quant = args.quant)

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
# for i, q in tqdm(enumerate(p_question)):
#     input_text = DP.get_prompt(prompt, cot, q)
#     answer, response, response_return = Model.parsing_yn(model, input_text, tokenizer, prompt)
#     print(f"Prompt {i}:")
#     print("The answer of this prompt is:", answer)
#     print("Original response from LLM", response_return)
#     print("Use this sentence to parse:", response)
#     if answer == "Parse fail":
#         fail += 1
#         continue
#     if answer == "Yes":
#         pos += 1
#         tp += 1
#     else:
#         fn += 1

# for i, q in tqdm(enumerate(n_question)):
#     input_text = DP.get_prompt(prompt, cot, q)
#     answer, response, response_return = Model.parsing_yn(model, input_text, tokenizer, prompt)
#     print(f"Prompt {i}:")
#     print("The answer of this prompt is:", answer)
#     print("Original response from LLM", response_return)
#     print("Use this sentence to parse:", response)
#     if answer == "Parse fail":
#         fail += 1
#         continue
#     if answer == "No":
#         pos += 1
#         tn += 1
#     else:
#         fp += 1

main_device = next(model.parameters()).device

for q in tqdm(p_question):
    input_text = DP.get_prompt(q)
    print(input_text)
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt").to(main_device)
        outputs = model(**inputs) # uncertainty
        logits = outputs.logits[:, -1, :]
        prob_distributions = F.softmax(logits, dim=-1)
        max_id = torch.argmax(prob_distributions).unsqueeze(0)
        print(max_id, prob_distributions[0, max_id])
        token_ = tokenizer.convert_ids_to_tokens(max_id)[0]
        print(token_)
        # 1. prob_dis 转换成dis最高的那个prob，然后找到对应的id，id再转token covert_ids_to_token
        # 2. 模型的yes和no怎么生成的，前面有没有分隔符，▁yes, ▁no, 找到gemma, llama, qwen的分隔符是什么
    
    prob_y = prob_distributions[0, token_y_list].sum()
    prob_n = prob_distributions[0, token_n_list].sum()
    print(prob_y, prob_n)
    if prob_y >= prob_n:
        pos += 1
        tp += 1
    else:
        fn += 1

for q in tqdm(n_question):
    input_text = DP.get_prompt(q)
    print(input_text)
    with torch.no_grad():
        inputs = tokenizer(input_text, return_tensors="pt").to(main_device)
        outputs = model(**inputs) # uncertainty
        logits = outputs.logits[:, -1, :]
        prob_distributions = F.softmax(logits, dim=-1)
        max_id = torch.argmax(prob_distributions).unsqueeze(0)
        print(max_id, prob_distributions[0, max_id])
        token_ = tokenizer.convert_ids_to_tokens(max_id)[0]
        print(token_)
    
    prob_y = prob_distributions[0, token_y_list].sum()
    prob_n = prob_distributions[0, token_n_list].sum()
    print(prob_y, prob_n)
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

# sys.exit()
# for q in tqdm(p_question) :
#     input_text = DP.get_prompt(prompt, cot, q)
#     with torch.no_grad():
#         input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#         hs = Model.get_hidden_states(model, tokenizer, input_text)
    
#     print(hs.shape)
#     hs_i = hs[:,-1,:]
#     prob_distributions = F.softmax(hs, dim=-1)
#     print("hs_i_rp_logs",hs_i.shape)
#     print(prob_distributions.shape)
#     try:
#         hs_i = hs_i.numpy()
#     except:
#         hs_i = hs_i.float().numpy()
    
#     print(hs_i.shape)
#     prob_y = hs_i[id_y1] + hs_i[id_y2] + hs_i[id_y3]
#     prob_n = hs_i[id_n1] + hs_i[id_n2] + hs_i[id_n3]

#     for i in range(tot_layer):
#         hs_i = hs[i,:,:]
#         # print("hs_i_rp_logs",hs_i.shape)
#         hs_i = hs_i[-1,:].cpu()
#         try:
#             hs_i = hs_i.numpy()
#         except:
#             hs_i = hs_i.float().numpy()
#         # print("hs_i_rp_logs",hs_i.shape)

#         rp_logs[i].append(hs_i)
#         torch.cuda.empty_cache()

# for q in tqdm(n_question) :
#     input_text = DP.get_prompt(prompt, cot, q)
#     with torch.no_grad():
#         input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#         hs = Model.get_hidden_states(model,tokenizer, input_text)
    
    
#     for i in range(tot_layer):
#         hs_i = hs[i,:,:]
#         # print("hs_i_rp_questions",hs_i.shape)

#         hs_i = hs_i[-1,:].cpu()
#         try:
#             hs_i = hs_i.numpy()
#         except:
#             hs_i = hs_i.float().numpy()
#         # print("hs_i_rp_questions",hs_i.shape)

#         rp_questions[i].append(hs_i)

# for i in range(tot_layer):
#     rp_log_data_list.append(np.array([tensor for tensor in rp_logs[i]]))
#     rp_question_data_list.append(np.array([tensor for tensor in rp_questions[i]]))

# labels_log = np.zeros(len(rp_log_data_list[0]))
# labels_question = np.ones(len(rp_question_data_list[0]))

# print("Evaluating ...")
# for i in range(tot_layer):
#     #print("i",i)
#     #rp_log_data_var_name = f'rp_log_data_{i}'
#     #rp_question_data_var_name = f'rp_question_data_{i}'
#     #rp_log_data_i = globals()[rp_log_data_var_name]
#     #rp_question_data_i = globals()[rp_question_data_var_name]
#     rp_log_data_i = rp_log_data_list[i]
#     rp_question_data_i = rp_question_data_list[i]

#     X = np.concatenate((rp_log_data_i, rp_question_data_i), axis=0)
#     y = np.concatenate((labels_log, labels_question), axis=0)
#     print(X) # 保存的是LLM输出来的答案
#     print(y)
    
#     # Merge data and labels
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#     # Initialize a random classifier, like Random Forest.
#     if args.clf == 'LR':
#         classifier = LogisticRegression(penalty='l2')
#     elif args.clf == 'RF':
#         classifier = RandomForestClassifier(random_state=42)
#     else:
#         raise ImportError("Cannot use the classifiers that the model haven't imported.")
#     classifier.fit(X_train, y_train)
#     y_pred = classifier.predict(X_test)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     print("------------------------------------")
#     print("This is epoch",i)
#     print(f'Accuracy: {accuracy}')

#     # Compute and print F1
#     # 'binary' for bi-classification problems; 'micro', 'macro' or 'weighted' for multi-classification problems
#     f1 = f1_score(y_test, y_pred, average='binary')  
#     print(f'F1 Score: {f1}')

#     # Predict the probability of test dataset. (For ROC AUC, we need probabilities instead of label)
#     y_prob = classifier.predict_proba(X_test)[:, 1]  # supposed + class is 1.

#     # Calc and print ROC, AUC
#     roc_auc = roc_auc_score(y_test, y_prob)
#     print(f'ROC AUC Score: {roc_auc}')

#     list_acc.append(accuracy)
#     list_f1.append(f1)
#     list_auc.append(roc_auc)

list_f1 = []
list_auc = []

# File saving and data
dict_res = {"Acc":list_acc, "F1":list_f1, "AUC": list_auc}
def LoadDataset(filename):
    with open(filename,'r+') as f:
        read_dict = f.read()
        f.close()
    read_dict = json.loads(read_dict)
    return read_dict

def SaveDataset(filename, dataset):
    dict_json = json.dumps(dataset)
    with open(filename,'w+') as f:
        f.write(dict_json)
        f.close()

model_name_refresh = {"google/gemma-7b":"gemma-7b", "google/gemma-2b": "gemma-2b", "meta-llama/Llama-2-7b-chat-hf": "Llama-7b", "meta-llama/Llama-2-13b-chat-hf":"Llama-13b", "meta-llama/Llama-2-70b-chat-hf":"Llama-70b","Qwen/Qwen1.5-0.5B":"Qwen-0.5B","Qwen/Qwen1.5-1.8B":"Qwen-1.8B","Qwen/Qwen1.5-4B":"Qwen-4B","Qwen/Qwen1.5-7B":"Qwen-7B",
                      "Qwen/Qwen1.5-14B":"Qwen-14B","Qwen/Qwen1.5-72B":"Qwen-72B", "meta-llama/Meta-Llama-3-8B": "Llama3-8B"}
model_name = model_name_refresh[args.model]

save_path_final = args.savepath + f"{args.dataset}_{model_name}_{args.quant}_{args.noise}_{args.clf}.json"
#SaveDataset(save_path_final, dict_res)
#print(list_acc)
#print(list_f1)
#print(list_auc)

print(f"Acc = {acc}, Fail Rate = {fr}")
print(f"TP = {tp}, TN = {tn}")
print(f"FP = {fp}, FN = {fn}")