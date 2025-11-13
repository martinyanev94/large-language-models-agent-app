pip install transformers
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Sample text
text = ["I love using BERT for natural language processing tasks."]

# Encoding the text and creating tensor inputs
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Making predictions with the model
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Process logits to determine class predictions
predictions = torch.argmax(logits, dim=-1)
print("Predicted class:", predictions.item())
