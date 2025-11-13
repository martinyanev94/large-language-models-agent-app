from transformers import Trainer, TrainingArguments

# Let's say you have a dataset prepared
train_dataset = ...  # Load or create your dataset here

# Specify training arguments
training_args = TrainingArguments(
    output_dir='./results',         
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
)

# Create a Trainer instance
trainer = Trainer(
    model=model,                     
    args=training_args,              
    train_dataset=train_dataset       
)

# Start the fine-tuning
trainer.train()
