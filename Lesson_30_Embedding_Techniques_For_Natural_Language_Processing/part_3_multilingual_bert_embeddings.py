from transformers import BertTokenizer, BertModel
import torch

# Load multilingual BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
model = BertModel.from_pretrained('bert-base-multilingual-uncased')

# Encode a sentence in different languages
input_text = "El perro ladró al cartero."
inputs = tokenizer(input_text, return_tensors="pt")

# Get outputs
with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state[:, 0, :]
print(embeddings)
