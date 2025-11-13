from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased')

# Encode text
input_text = "The cat sat on the mat."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)

# The last hidden state is the output from the model
last_hidden_states = outputs.last_hidden_state
