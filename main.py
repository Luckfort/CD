from dataset import DataProcessing
import json
from utils import add_noise, LLM
import argparse
import torch
#from huggingface_hub import notebook_login
#notebook_login()
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

parser = argparse.ArgumentParser()

parser.add_argument('--savepath', type=str, default="../lab_rs1/", help="Path for save Result")
parser.add_argument('--dataset', type=str, default='STSA', help="Dataset")
parser.add_argument('--datapath', type=str, default='./dataset/stsa.binary.train', help="Default data path")
parser.add_argument('--model', type=str, default='google/gemma-7b', help="Which LLM to use")
parser.add_argument('--cuda', type=int, default=0, help="Cuda ID")
parser.add_argument('--quant', type=int, default=32, help="Quantization")
parser.add_argument('--noise', type=str, default='non-noise', help="Whether to add noise")
parser.add_argument('--clf', type=str, default='LR', help="Classifier")

args = parser.parse_args()
if args.quant != 32:
    if args.quant == 8:
        quantization_config1 = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quant == 16:
        quantization_config1 = BitsAndBytesConfig(load_in_16bit=True)
    else:
        raise ValueError("We don't have this quantization bit! Please try 8, 16, 32.")

tokenizer = AutoTokenizer.from_pretrained(args.model)
if args.quant != 32:
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config = quantization_config1)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model)

file_path = args.datapath

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm import tqdm

model_2_layer = {"google/gemma-7b": 28, "google/gemma-2b": 18, "meta-llama/Llama-2-7b-chat-hf": 32}
tot_layer = model_2_layer[args.model]

DP = DataProcessing(data_path=args.datapath, data_name=args.dataset, noise=args.noise)

# The prompts are different among the datasets!!!
p_question, n_question, r_question, prompt, cot = DP.dispacher()

Model = LLM(cuda_id = args.cuda, layer_num = tot_layer, quant = args.quant)

print("Model is training ...")
list_acc = []
list_f1 = []
list_auc = []
prompt = 'According to the sentence, judge the emotion is Positive or Negative.'

rp_questions_dic =[]
rp_log_dic = []
rp_logs = [[] for _ in range(tot_layer)]
rp_questions = [[] for _ in range(tot_layer)]
rp_others = [[] for _ in range(tot_layer)]

rp_log_data_list = []
rp_question_data_list = []
rp_other_data_list = []

for q in tqdm(p_question) :
    input_text = DP.get_prompt(prompt, cot, q)
    with torch.no_grad():
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        hs = Model.get_hidden_states(model, tokenizer, input_text)
    for i in range(tot_layer):
        hs_i = hs[i,:,:]
        # print("hs_i_rp_logs",hs_i.shape)
        hs_i = hs_i[-1,:].cpu()
        hs_i = hs_i.numpy()
        # print("hs_i_rp_logs",hs_i.shape)

        rp_logs[i].append(hs_i)
        #torch.cuda.empty_cache()

for q in tqdm(n_question) :
    input_text = DP.get_prompt(prompt, cot, q)
    with torch.no_grad():
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        hs = Model.get_hidden_states(model,tokenizer, input_text)
    for i in range(tot_layer):
        hs_i = hs[i,:,:]
        # print("hs_i_rp_questions",hs_i.shape)

        hs_i = hs_i[-1,:].cpu()
        hs_i = hs_i.numpy()
        # print("hs_i_rp_questions",hs_i.shape)

        rp_questions[i].append(hs_i)

if args.dataset == 'hateeval':
    for q in tqdm(r_question) :
        input_text = DP.get_prompt(prompt, cot, q)
        with torch.no_grad():
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            hs = Model.get_hidden_states(model,tokenizer, input_text)
        for i in range(tot_layer):
            hs_i = hs[i,:,:]
            # print("hs_i_rp_questions",hs_i.shape)

            hs_i = hs_i[-1,:].cpu()
            hs_i = hs_i.numpy()
            # print("hs_i_rp_questions",hs_i.shape)

            rp_others[i].append(hs_i)

for i in range(tot_layer):
    rp_log_data_list.append(np.array([tensor for tensor in rp_logs[i]]))
    rp_question_data_list.append(np.array([tensor for tensor in rp_questions[i]]))
    if args.dataset == 'hateeval':
        rp_other_data_list.append(np.array([tensor for tensor in rp_others[i]]))

labels_log = np.zeros(len(rp_log_data_list[0]))
labels_question = np.ones(len(rp_question_data_list[0]))
if args.dataset == 'hateeval':
    labels_other = int(2) * np.ones(len(rp_question_data_list[0]))

print("Evaluating ...")
for i in range(tot_layer):
    #print("i",i)
    #rp_log_data_var_name = f'rp_log_data_{i}'
    #rp_question_data_var_name = f'rp_question_data_{i}'
    #rp_log_data_i = globals()[rp_log_data_var_name]
    #rp_question_data_i = globals()[rp_question_data_var_name]
    rp_log_data_i = rp_log_data_list[i]
    rp_question_data_i = rp_question_data_list[i]
    if args.dataset == 'hateeval':
        rp_other_data_i = rp_other_data_list[i]

    if args.dataset != 'hateeval':
        X = np.concatenate((rp_log_data_i, rp_question_data_i), axis=0)
        y = np.concatenate((labels_log, labels_question), axis=0)
    else:
        X = np.concatenate((rp_log_data_i, rp_question_data_i, rp_other_data_i), axis=0)
        y = np.concatenate((labels_log, labels_question, labels_other), axis=0)
    # Merge data and labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Initialize a random classifier, like Random Forest.
    if args.clf == 'LR':
        classifier = LogisticRegression(penalty='l2')
    elif args.clf == 'RF':
        classifier = RandomForestClassifier(random_state=42)
    else:
        raise ImportError("Cannot use the classifiers that the model haven't imported.")
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("------------------------------------")
    print("This is epoch",i)
    print(f'Accuracy: {accuracy}')

    # Compute and print F1
    # 'binary' for bi-classification problems; 'micro', 'macro' or 'weighted' for multi-classification problems
    if args.dataset == 'hateeval':
        f1 = f1_score(y_test, y_pred, average='macro')  
    else:
        f1 = f1_score(y_test, y_pred, average='binary')  
    print(f'F1 Score: {f1}')

    # Predict the probability of test dataset. (For ROC AUC, we need probabilities instead of label)
    y_prob = classifier.predict_proba(X_test)[:, 1]  # supposed + class is 1.

    # Calc and print ROC, AUC
    if args.dataset == 'hateeval':
        roc_auc = roc_auc_score(y, y_prob, average='macro', multi_class='ovo',labels=[0, 1, 2])
    else:
        roc_auc = roc_auc_score(y_test, y_prob)
    print(f'ROC AUC Score: {roc_auc}')

    
    list_acc.append(accuracy)
    list_f1.append(f1)
    list_auc.append(roc_auc)

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

model_name_refresh = {"google/gemma-7b":"gemma-7b", "google/gemma-2b": "gemma-2b", "meta-llama/Llama-2-7b-chat-hf": "Llama-7b"}
model_name = model_name_refresh[args.model]

save_path_final = args.savepath + f"{args.dataset}_{model_name}_{args.quant}_{args.noise}_{args.clf}.json"
SaveDataset(save_path_final, dict_res)
print(list_acc)
print(list_f1)
print(list_auc)