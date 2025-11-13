from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode a sentence
sentence = "The bank can guarantee your bank's safety."
inputs = tokenizer(sentence, return_tensors='pt')
outputs = model(**inputs)

# Get the embeddings for the tokens
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
