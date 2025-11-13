from transformers import BertTokenizer, BertModel
import torch

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Sample sentence
sentence = "I love programming in Python"
inputs = tokenizer(sentence, return_tensors="pt")

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)

# The last hidden states represent the embeddings
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
