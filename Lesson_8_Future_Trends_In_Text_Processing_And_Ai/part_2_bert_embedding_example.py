from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode input text
input_text = "The quick brown fox jumps over the lazy dog."
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Get embeddings
with torch.no_grad():
    outputs = model(input_ids)

# Extracting the last hidden states
last_hidden_states = outputs.last_hidden_state
print("Shape of last hidden states:", last_hidden_states.shape)
