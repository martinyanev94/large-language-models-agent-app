from transformers import BertTokenizer, BertModel
import torch

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode sentences
inputs = tokenizer(documents, return_tensors='pt', padding=True, truncation=True)

# Get BERT embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

# Display embeddings for the first document
print("\nBERT Embeddings for Document 1:\n", embeddings[0][0])
