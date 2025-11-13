# Load a simple news classification dataset
dataset_classification = load_dataset("AG_NEWS")
model_classification = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# Tokenize the classification dataset
tokenized_classification_dataset = dataset_classification.map(preprocess_function, batched=True)

# Define training arguments for classification
training_args_classification = TrainingArguments(
    output_dir='./classification_results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

# Create a Trainer instance for classification
trainer_classification = Trainer(
    model=model_classification,
    args=training_args_classification,
    train_dataset=tokenized_classification_dataset['train'],
    eval_dataset=tokenized_classification_dataset['test']
)

# Start training the classification model
trainer_classification.train()
