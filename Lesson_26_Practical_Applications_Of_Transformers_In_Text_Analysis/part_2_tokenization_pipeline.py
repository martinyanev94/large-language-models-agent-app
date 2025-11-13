def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

# Prepare the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)
