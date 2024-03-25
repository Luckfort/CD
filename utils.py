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

    def get_hidden_states(self, model, tok, prefix, device="cuda:1"):
        device = f"cuda:{self.cuda_id}"
        if self.quant == 32:
            inp = {k: torch.tensor(v)[None].to(f"cuda:{self.cuda_id}") for k, v in tok(prefix).items()}
            model = model.to(f"cuda:{self.cuda_id}")
            with TraceDict(model, self.layer_names) as tr:
                logits = model(**inp)['logits']
            return torch.stack(
                [tr[ln].output[0][None, :] if ln == "transformer.wte" else tr[ln].output[0] for ln in self.layer_names])
        else:
            inp = {k: torch.tensor(v)[None] for k, v in tok(prefix).items()}
            with TraceDict(model, self.layer_names) as tr:
                logits = model(**inp)['logits']
            return torch.stack(
                [tr[ln].output[0][None, :] if ln == "transformer.wte" else tr[ln].output[0] for ln in self.layer_names])
    
        
