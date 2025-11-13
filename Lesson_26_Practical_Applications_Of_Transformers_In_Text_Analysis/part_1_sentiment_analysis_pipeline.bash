pip install transformers
pip install torch
pip install datasets
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

# Load pre-trained tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load a sentiment analysis dataset, for instance, the IMDB dataset
dataset = load_dataset("imdb")
