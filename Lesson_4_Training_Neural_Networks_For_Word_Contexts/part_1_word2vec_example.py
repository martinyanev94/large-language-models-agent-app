!pip install gensim
import gensim
from gensim.models import Word2Vec

# Sample dataset
sentences = [
    ["the", "bank", "will", "not", "lend", "you", "money"],
    ["he", "sat", "on", "the", "bank", "of", "the", "river"],
    ["many", "people", "enjoy", "fishing", "at", "the", "river"],
    ["people", "need", "to", "manage", "their", "finances"],
    ["the", "financial", "institution", "is", "always", "busy"]
]
