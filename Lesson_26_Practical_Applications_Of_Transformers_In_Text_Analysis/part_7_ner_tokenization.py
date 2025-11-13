dataset_ner = load_dataset("conll2003")
model_ner = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=9)  # Adjust num_labels as needed
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []

    for i in range(len(tokenized_inputs["input_ids"])):
        # Create label list aligned to tokenized inputs
        label = [-100] * len(tokenized_inputs["input_ids"][i])  # Initialize with -100 (ignore index)
        for j in range(len(examples['ner_tags'][i])):
            if tokenized_inputs["input_ids"][i][j] != tokenizer.pad_token_id:
                label[j] = examples['ner_tags'][i][j]
        labels.append(label)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize the NER dataset
tokenized_ner_dataset = dataset_ner.map(tokenize_and_align_labels, batched=True)
