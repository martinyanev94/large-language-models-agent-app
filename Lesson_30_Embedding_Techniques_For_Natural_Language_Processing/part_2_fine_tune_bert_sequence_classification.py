from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load your dataset
dataset = load_dataset('your_custom_dataset')

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

# Trainer to handle training and evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

# Start fine-tuning
trainer.train()
