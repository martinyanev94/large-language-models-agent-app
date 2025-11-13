# Get the embeddings for the specific tokens
bank_embedding = last_hidden_states[0][inputs['input_ids'][0][2]]  # Index 2 corresponds to 'bank'
print(bank_embedding)
