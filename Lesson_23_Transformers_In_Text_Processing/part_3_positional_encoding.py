def positional_encoding(max_len, embed_dim):
    pos = np.arange(max_len)[:, np.newaxis]  # Shape (max_len, 1)
    i = np.arange(embed_dim)[np.newaxis, :]  # Shape (1, embed_dim)
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(embed_dim))
    angle_rads = pos * angle_rates
    # Apply sine to even index and cosine to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads

max_len = 10  # Define the maximum length of the sequence
embed_dim = 64  # Define the embedding dimension
pos_enc = positional_encoding(max_len, embed_dim)

print("Positional Encodings:")
print(pos_enc)
