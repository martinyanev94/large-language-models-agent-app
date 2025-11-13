pip install transformers
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample text
text = "The bank can guarantee deposits and loans."
inputs = tokenizer(text, return_tensors='pt')

# Get the outputs
with torch.no_grad():
    outputs = model(**inputs)

# Embedding for the word 'bank'
bank_index = inputs['input_ids'][0].tolist().index(tokenizer.encode('bank')[1])  # Finding 'bank' in tokens
word_embedding = outputs.last_hidden_state[0][bank_index]
print("Contextual embedding for 'bank':", word_embedding)
