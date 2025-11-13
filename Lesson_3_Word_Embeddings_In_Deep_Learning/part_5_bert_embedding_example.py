!pip install transformers
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Tokenize input
inputs = tokenizer("the bank can lend money", return_tensors="pt")

# Perform a forward pass to get embeddings
with torch.no_grad():
    outputs = model(**inputs)

# Take the output from the last hidden state
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
