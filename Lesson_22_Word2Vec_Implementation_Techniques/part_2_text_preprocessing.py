import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Sample text data
corpus = [
    "Natural language processing is an exciting field.",
    "Word embeddings capture semantic relationships between words.",
    "The quick brown fox jumps over the lazy dog."
]

# Prepare stop words
stop_words = set(stopwords.words('english'))

# Tokenization and preprocessing
preprocessed_corpus = []
for sentence in corpus:
    tokens = word_tokenize(sentence.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    preprocessed_corpus.append(filtered_tokens)

print(preprocessed_corpus)
