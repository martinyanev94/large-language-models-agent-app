from transformers import BertForSequenceClassification, Trainer, TrainingArguments

# Load pre-trained BERT for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

# Trainer setup and model training would follow
