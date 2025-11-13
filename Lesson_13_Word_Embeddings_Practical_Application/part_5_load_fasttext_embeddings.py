from gensim.models import FastText

# Load pre-trained FastText model for multilingual embeddings
fasttext_model = FastText.load_fasttext_format('path_to_multilingual_model.bin')

# Get embedding for the word "hello" in English
english_word = fasttext_model.wv['hello']
print(f"Embedding for 'hello': {english_word}")
