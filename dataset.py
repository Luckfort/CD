from utils import add_noise
import pandas as pd
import json
from datasets import load_dataset

class DataProcessing():
    
    def __init__(self, data_path, data_name, noise):
        self.data_path = data_path
        self.data_name = data_name
        self.noise = noise
    
    """
        Select the way to get p and n samples given the dataset.
    """
    def dispacher(self):
        cot = None
        if self.data_name == 'StrategyQA':
            p, q = self.StrategyQA()
            prompt = 'Judge whether the answer of the question is true.  Please answer the question in the following format: first say "Yes" or "No", then tell the reasons. Answer concisely.'  + '\n' \
                'Q: Will Queen Elizabeth be buried in the Pantheon?' + '\n' \
                "Let us think step by step. The stem of the sentence is Queen Elizabeth, burial, pantheon. Inference: First, the Pantheon is a church, so it is possible that she could be buried there. Second, Queen Elizabeth II is still alive, so she has not been buried yet. Third, even if she were to be buried in the Pantheon, it is unlikely that we would know about it ahead of time, so it is hard to say for sure." + '\n' \
                'pred_ans: no'
            cot = 'Let us think step by step...'
            
        elif self.data_name == 'coinflip':
            p, q = self.coinflip()
            prompt = 'According to the flipping process above, determine if a coin remains heads up after it is either flipped or left unflipped by individuals. Please answer the question in the following format: first say "Yes" or "No", then tell the reasons. Answer concisely.'
            
        elif self.data_name == 'cities':
            p, q = self.cities()
            prompt = 'Judge whether the statement is True or not. Your answer should be in the following format. Answer concisely. Answer: {Yes or No}. Reason: {Why you choose that answer}'
            
        elif self.data_name == 'common':
            p, q = self.common()
            prompt = 'Judge whether the statement is True or not. Your answer should be in the following format. Answer concisely. Answer: {Yes or No}. Reason: {Why you choose that answer}'
            
        elif self.data_name == 'counterfact':
            p, q = self.counterfact()
            prompt = 'Judge whether the statement is true or not. Your answer should be in the following format. Answer concisely. Answer: {Yes or No}. Reason: {Why you choose that answer}'
        
        elif self.data_name == 'hateeval':
            p, q = self.hateeval()
            prompt = 'According to the comment, tell whether they present hate speech or not. Please answer the question in the following format: first say "Yes" or "No", then tell the reasons. Answer concisely.'
            
        elif self.data_name == 'STSA':
            p, q = self.STSA()
            prompt = 'The sentence above is a movie review and reflects the writer\'s overall intention for this review. According to the sentence, judge whether the emotion is Positive or Negative. If Positive, please say yes ahead of the answer. If Negative, please say no ahead of the answer. Please answer the question in the following format: first say "Yes" or "No", then tell the reasons. Answer concisely.'
            
        elif self.data_name == 'IMDb':
            p, q = self.IMDb()
            prompt = 'According to the movie review, judge whether it is Positive or Negative. If Positive, please say yes ahead of the answer. If Negative, please say no ahead of the answer. Please answer the question in the following format: first say "Yes" or "No", then tell the reasons. Answer concisely.'
            
        elif self.data_name == 'sarcasm':
            p, q = self.sarcasm()
            prompt = 'Task: Detect sarcasm, help me identify whether this sentence is sarcastic.' + '\n' \
                    'First, we need to understand what sarcasm is. Sarcasm is a form of verbal irony, '+ '\n' \
                    'where the intended meaning of the words is the opposite of the literal meaning. '+ '\n' \
                    'In other words, the speaker is saying one thing but meaning the opposite. '
            cot = 'Think carefully according to the sentence. Is there any sarcasm in this sentence? Please answer the question in the following format: first say "Yes" or "No", then tell the reasons. Answer concisely.'   
        
        return p, q, prompt, cot
    
    """
        return the context that we enter to LLM.
    """
    def get_prompt(self, prompt, cot, question):
        if self.data_name in ['common','cities','counterfact']:
            new_prompt = prompt + " " + question
        elif self.data_name in ['STSA','coinflip','IMDb','hateeval']:
            new_prompt = question + " " + prompt
        elif self.data_name in ['sarcasm','StrategyQA']:
            new_prompt = prompt + question + " " + cot
        
        return new_prompt
    
    def STSA(self):
        p_question = []
        n_question = []
        with open(self.data_path, 'r') as file:
            for line in file:
                line = line.strip()
                parts = line.split(' ', 1)
                label = int(parts[0])
                question = parts[1]
                if self.noise == 'noise':
                    question = add_noise(question)
                if label == 0:
                    #question = parts[1] 
                    p_question.append(question)
                else:
                    #question = parts[1] 
                    n_question.append(question)
        return p_question, n_question

    def StrategyQA(self):
        yes_inputs = []
        no_inputs = []

        json_file_path = './dataset/StrategyQA_task.json'
            
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            for example in data['examples']:
                if example['target_scores']['Yes'] == 1: 
                    
                    line = example['target'] 
                    if line.startswith("Yes."):
                        line = line[4:].strip()
                    elif line.startswith("No."):
                        line = line[3:].strip()
                    target = example['input']+" "+line
                    if self.noise == 'noise':
                        target = add_noise(target)
                    yes_inputs.append(target)
                    
                elif example['target_scores']['No'] == 1:
                    line = example['target'] 
                    if line.startswith("Yes."):
                        line = line[4:].strip()
                    elif line.startswith("No."):
                        line = line[3:].strip()
                    
                    target = example['input']+" "+line
                    if self.noise == 'noise':
                        target = add_noise(target)
                    no_inputs.append(target)
        
        return yes_inputs, no_inputs
    
    def coinflip(self):
        json_file_path = './dataset/coin_flip.json'
        with open(json_file_path, 'r') as file:
            data1 = json.load(file)
        p_question = []
        n_question = []
        for i in data1['examples']:
            a = i['question']
            if self.noise == 'noise':
                a = add_noise(a)
            b = i['answer']
            if(b=='yes'):
                p_question.append(a)
            if(b=='no'):
                n_question.append(a)
    
        return p_question, n_question
    
    def cities(self):
        csv_file_path = './dataset/cities.csv'
        data = pd.read_csv(csv_file_path)
        # Filter the statements based on the label and get the respective lists
        lista = data[data['label'] == 1]['statement'].tolist()
        listb = data[data['label'] == 0]['statement'].tolist()
        
        list_true_noise = []
        list_false_noise = []
        for i in range(len(lista)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(lista[i]))
            else:
                list_true_noise.append(lista[i])
        
        for i in range(len(listb)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(listb[i]))
            else:
                list_false_noise.append(listb[i])
        
        return list_true_noise, list_false_noise
        #return list_true_noise[:5], list_false_noise[:5]
    
    def common(self):
        csv_file_path = './dataset/common_claim.csv'
        data = pd.read_csv(csv_file_path)
        list_true = data[data['label'] == 'True' ]['examples'].tolist()
        list_false = data[data['label'] == 'False' ]['examples'].tolist()
        
        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true[i]))
            else:
                list_true_noise.append(list_true[i])
        
        for i in range(len(list_false)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false[i]))
            else:
                list_false_noise.append(list_false[i])
        
        return list_true_noise[:3000], list_false_noise[:3000]
    
    def counterfact(self):
        csv_file_path = './dataset/counterfact.csv'
        data = pd.read_csv(csv_file_path)
        list_true1 = data[data['label'] == 1]['statement'].tolist()
        list_false1 = data[data['label'] == 0]['statement'].tolist()
        
        list_true_noise = []
        list_false_noise = []
        for i in range(len(list_true1)):
            if self.noise == 'noise':
                list_true_noise.append(add_noise(list_true1[i]))
            else:
                list_true_noise.append(list_true1[i])
        
        for i in range(len(list_false1)):
            if self.noise == 'noise':
                list_false_noise.append(add_noise(list_false1[i]))
            else:
                list_false_noise.append(list_false1[i])
        
        return list_true_noise[:2000], list_false_noise[:2000]
    
    def hateeval(self):
        df = pd.read_csv('./dataset/hateeval.tsv', sep='\t')
        list1 = []
        list2 = []
        list3 = []
        for i,a,b,c in zip(df['text'],df['HS'],df['TR'],df['AG']):
            if self.noise == 'noise':
                i = add_noise(i)
            if a== 1 :
                list1.append(i)
            else :
                list2.append(i)
        return list1[:3000], list2[:3000]
    
    def IMDb(self):
        dataset = load_dataset("imdb")
        list1 = []
        list2 = []
        for i,t in zip(dataset['test']['text'],dataset['test']['label']):
            if self.noise == 'noise':
                i = add_noise(i)
            if t==0:
                list1.append(i)
            if t==1:
                list2.append(i)
        
        return list1[:1000], list2[:1000]
    
    def sarcasm(self):
        json_file_path = './dataset/sarcasm.json'

        sarcastic_headlines = []
        non_sarcastic_headlines = []

        # Read the JSON Lines file
        with open(json_file_path, 'r') as file:
            for line in file:
                entry = json.loads(line)
                entry_headline = entry['headline']
                if self.noise == 'noise':
                    entry_headline = add_noise(entry_headline)
                
                # Extract headlines into the appropriate lists based on sarcasm label
                if entry['is_sarcastic'] == 1:
                    sarcastic_headlines.append(entry_headline)
                else:
                    non_sarcastic_headlines.append(entry_headline)
        
        return sarcastic_headlines[:3000], non_sarcastic_headlines[:3000]