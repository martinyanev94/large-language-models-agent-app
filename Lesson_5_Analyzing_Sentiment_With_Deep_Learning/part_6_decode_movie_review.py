import numpy as np

def decode_review(encoded_review):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    return decoded_review

# Test on a sample review
sample_review = x_test[0]
print(decode_review(sample_review))  # Decoding the review
print("Predicted sentiment (0: Negative, 1: Positive):", np.round(model.predict(sample_review.reshape(1, max_len))))
