from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Sample input 
texts = ["I love watching movies.", "I dislike waiting in lines."]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Make predictions
with torch.no_grad():
    outputs = model(**inputs)

print(outputs)
