training_args_ner = TrainingArguments(
    output_dir='./ner_results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

trainer_ner = Trainer(
    model=model_ner,
    args=training_args_ner,
    train_dataset=tokenized_ner_dataset['train'],
    eval_dataset=tokenized_ner_dataset['test']
)

# Start training NER model
trainer_ner.train()
