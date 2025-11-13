import numpy as np

def self_attention(input_seq):
    attention_weights = np.dot(input_seq, input_seq.T)
    attention_weights = attention_weights / np.sqrt(input_seq.shape[-1])  # Scale
    attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights), axis=1, keepdims=True)
    output = np.dot(attention_weights, input_seq)
    return output, attention_weights

# Example input sequence
input_sequence = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Simple 3x3 representation
output_seq, weights = self_attention(input_sequence)

print("Output Sequence:")
print(output_seq)
print("Attention Weights:")
print(weights)
