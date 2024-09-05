import re
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, baukit
from baukit import TraceDict
import re
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

"""
    Adding noise to the questions.
    Input: 
        question - the question needs to add noise.
    Return:
        noised_question - the question after adding noise.
    should have only aaa/bbb
"""
import numpy as np
def add_noise(question):
    # Adding three letters after the period.
    noise = np.random.choice(a=2) + 97
    noise = [chr(noise)] * 3
    noised_question = ''.join(noise) + question
    return noised_question

class LLM(torch.nn.Module):
    
    """
        Given a total layer, we construct a layer name list.
    """
    def __init__(self, cuda_id, layer_num, quant):
        self.layer_num = layer_num
        self.cuda_id = cuda_id
        self.layer_names = []
        self.quant = quant
        for i in range(self.layer_num):
            self.layer_names.append(f'model.layers.{i}.post_attention_layernorm')

    def get_hidden_states(self, model, tok, prefix, device="cuda:1", accelerator = None):
        device = f"cuda:{self.cuda_id}"
        if self.quant == 32:
            inp = {k: torch.tensor(v)[None].to(accelerator.device) for k, v in tok(prefix).items()}
            model = model.to(accelerator.device)
            with TraceDict(model, self.layer_names) as tr:
                logits = model(**inp)['logits']
            return torch.stack(
                [tr[ln].output[0][None, :] if ln == "transformer.wte" else tr[ln].output[0] for ln in self.layer_names])
            # inp = {k: torch.tensor(v)[None].to(f"cuda:{self.cuda_id}") for k, v in tok(prefix).items()}
            # model = model.to(f"cuda:{self.cuda_id}")
            # with TraceDict(model, self.layer_names) as tr:
            #     logits = model(**inp)['logits']
            # return torch.stack(
            #     [tr[ln].output[0][None, :] if ln == "transformer.wte" else tr[ln].output[0] for ln in self.layer_names])
        else:
            inp = {k: torch.tensor(v)[None] for k, v in tok(prefix).items()}
            with TraceDict(model, self.layer_names) as tr:
                logits = model(**inp)['logits']
            return torch.stack(
                [tr[ln].output[0][None, :] if ln == "transformer.wte" else tr[ln].output[0] for ln in self.layer_names])
    
    def parsing_yn(self, model, question, tokenizer, prompt, accelerator = None):
        # Encode the input question
        #inputs = tokenizer(question, return_tensors="pt").to(f"cuda:{self.cuda_id}")
        main_device = next(model.parameters()).device
        inputs = tokenizer(question, return_tensors="pt").to(main_device)
        #model = model.to(f"cuda:{self.cuda_id}")
        #model = model.to(accelerator.device)
        # Generate a response
        input_token_count = inputs['input_ids'].shape[-1]
        max_length = input_token_count + 64
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length = max_length)

        # Decode the output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        response_return = response
        response_all = response.replace(prompt, '')
        response_all = response_all.strip('\n').split('\n')
        response = response_all[-1]
        # Simple logic to classify the response
        
        if ("yes." in response.lower()) or ("yes," in response.lower()) or ("yes " in response.lower()):
            return "Yes", response, response_return
        elif ("no." in response.lower()) or ("no," in response.lower()) or ("no " in response.lower()):
            return "No", response, response_return # 用15个LLM来跑。gemma-2b, gemma-7b算2个, qwen-2做起 14B (LLaMA3-8B第一个跑)
        else:
            return "Parse fail", response, response_return
