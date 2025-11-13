class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(num_heads, embed_dim)
        self.attention2 = MultiHeadAttention(num_heads, embed_dim)
        self.ffn = self.feed_forward_network(embed_dim, ff_dim)

    def feed_forward_network(self, embed_dim, ff_dim):
        return tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'), 
            tf.keras.layers.Dense(embed_dim)
        ])

    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1 = self.attention1(x, x, x, look_ahead_mask)
        attn2 = self.attention2(enc_output, enc_output, attn1, padding_mask)
        output = self.ffn(attn2)
        return output
